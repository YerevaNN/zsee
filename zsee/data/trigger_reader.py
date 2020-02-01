import logging

import pickle

from abc import ABC
from typing import List, Dict, Tuple, Union, Iterable, cast

from overrides import overrides

from allennlp.data import DatasetReader, Instance, Field, TokenIndexer, Token, Tokenizer
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField, MultiLabelField
from spacy.tokens import Token as SpacyToken

from .translation_service import TranslationService

logger = logging.getLogger(__name__)


class TriggerReader(DatasetReader, ABC):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 trigger_label_namespace: str = 'event_labels',
                 multi_label: bool = True,
                 null_label: bool = False,
                 translation_service: TranslationService = None,
                 tokenizer: Tokenizer = None,
                 filter_instances: bool = False,
                 sentence_level_only: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)

        self._token_indexers = token_indexers
        self._trigger_label_namespace = trigger_label_namespace
        self._multi_label = multi_label
        self._null_label = null_label
        self._translation_service = translation_service
        self._filter = filter_instances
        self._sentence_level_only = sentence_level_only

        # TODO temporary hack
        if tokenizer is None:
            if hasattr(translation_service, "tokenizer"):
                tokenizer = translation_service.tokenizer
            else:
                tokenizer = ...

        self._tokenizer = tokenizer

    @overrides
    def read(self, file_path: str) -> Iterable[Instance]:
        instances = super().read(file_path)
        # Print some statistics here
        num_samples = len(instances)
        logger.warning(f'Instances read: {num_samples}')
        num_multi_labels_samples = 0
        for instance in instances:
            field = cast(MultiLabelField, instance['sentence_trigger_labels'])
            if len(field.labels) > 1:
                num_multi_labels_samples += 1
        logger.warning(f'Multi-label samples: {num_multi_labels_samples}, {num_multi_labels_samples / num_samples * 100}')

        return instances

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        for instance in self._read_instances(file_path):
            if instance is None:
                continue
            yield instance

    def _read_instances(self, file_path: str) -> Iterable[Instance]:
        raise NotImplementedError

    def _build_instance(self,
                        tokens: List[Token],
                        trigger_labels: List[str] = None,
                        trigger_token_seqs: Dict[Tuple[int, int], str] = None,
                        **metadata) -> Instance:
        if len(tokens) < 3 and self._filter is True:
            return None

        # Translate sentence if translation_service is provided
        if self._translation_service is not None:
            source_snt = self._detokenize(tokens)
            target_snt = self._translation_service(source_snt)
            tokens = self._tokenize(target_snt)

        fields: Dict[str, Field] = dict()

        # First, populate fields with provided metadata
        for key, value in metadata.items():
            fields[key] = MetadataField(value)

        if trigger_token_seqs is not None:
            fields['trigger_token_seqs'] = MetadataField(trigger_token_seqs)

        # Building different discrete representations for text embedders.
        text_field = TextField(tokens, self._token_indexers)
        fields['text'] = text_field
        # Additionally, raw tokens are also stored for reverse mapping
        fields['tokens'] = MetadataField(tokens)

        # Build an Instance without annotations to use in inference phase.
        if trigger_labels is None:
            return Instance(fields)

        if self._translation_service is None:
            # If the sentence is translated we have no alignments
            # for token-level labels, so we skip them

            if len(trigger_labels) > len(tokens):
                truncate_len = len(tokens)
                trigger_labels = trigger_labels[:truncate_len]
                logger.warning('Truncated tokens detected. Truncating labels as well.')

            # Token-level trigger labels
            trigger_labels_field = SequenceLabelField(trigger_labels,
                                                      text_field,
                                                      self._trigger_label_namespace)
            if not self._sentence_level_only:
                fields['trigger_labels'] = trigger_labels_field

        # Sentence-level trigger label(s)
        # if not self._multi_label:
        #     raise NotImplementedError

        token_tags = set(trigger_labels)
        sentence_trigger_labels = [tag for tag in token_tags if tag != 'O']
        if not sentence_trigger_labels and self._null_label:
            sentence_trigger_labels = ['O']
        fields['sentence_trigger_labels'] = MultiLabelField(sentence_trigger_labels,
                                                            self._trigger_label_namespace)
        return Instance(fields)

    def string_to_instance(self, text: str) -> Instance:
        tokens = self._tokenize(text)
        return self.tokens_to_instance(tokens)

    def tokens_to_instance(self, tokens: List[Union[str, Token, SpacyToken]]) -> Instance:
        # Just in case if model needs raw sentence for decoding
        tokens_with_whitespaces = [
            getattr(token, 'text_with_ws', f'{token} ')  # use non-destructive tokens
            for token in tokens                               # if available
        ]
        raw_sentence = ''.join(tokens_with_whitespaces).rstrip(' ')
        return self._build_instance(tokens, raw_sentence=raw_sentence)

    @overrides
    def text_to_instance(self, text: Union[str, List[Union[str, Token, SpacyToken]]]) -> Instance:
        if isinstance(text, str):
            return self.string_to_instance(text)
        else:
            return self.tokens_to_instance(text)

    @overrides
    def _instances_from_cache_file(self, cache_filename: str) -> Iterable[Instance]:
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        with open(cache_filename, 'wb') as f:
            pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _tokenize(self, sentence: str) -> List[Token]:
        return self._tokenizer.tokenize(sentence)

    def _detokenize(self, tokens: List[Token]) -> str:
        tokens_with_whitespaces = [
            getattr(token, 'text_with_ws', f'{token} ')  # use non-destructive tokens
            for token in tokens                          # if available
        ]
        return ''.join(tokens_with_whitespaces).rstrip(' ')
