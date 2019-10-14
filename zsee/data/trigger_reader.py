import logging

from abc import ABC
from typing import List, Dict, Tuple, Union

from allennlp.data import DatasetReader, Instance, Field, TokenIndexer, Token
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField
from spacy.tokens import Token as SpacyToken

logger = logging.getLogger(__name__)


class TriggerReader(DatasetReader, ABC):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 trigger_label_namespace: str,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

        self._token_indexers = token_indexers
        self._trigger_label_namespace = trigger_label_namespace

    def _build_instance(self,
                        tokens: List[Token],
                        trigger_labels: List[str] = None,
                        trigger_token_seqs: Dict[Tuple[int, int], str] = None,
                        **metadata) -> Instance:
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

        if len(trigger_labels) > len(tokens):
            truncate_len = len(tokens)
            trigger_labels = trigger_labels[:truncate_len]
            logger.warning('Truncated tokens detected. Truncating labels as well.')

        trigger_labels_field = SequenceLabelField(trigger_labels,
                                                  text_field,
                                                  self._trigger_label_namespace)
        fields['trigger_labels'] = trigger_labels_field

        return Instance(fields)

    def text_to_instance(self, tokens: List[Union[str, Token, SpacyToken]]) -> Instance:
        # Just in case if model needs raw sentence for decoding
        raw_sentence = ''.join([
            getattr(token, 'text_with_ws', str(token))  # use non-destructive tokens
            for token in tokens                         # if available
        ])
        return self._build_instance(tokens, raw_sentence=raw_sentence)
