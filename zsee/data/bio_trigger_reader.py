import logging
from typing import List, Dict, Tuple, Iterator

from tqdm import tqdm

from allennlp.data import Token, Instance, TokenIndexer, DatasetReader
from .translation_service import TranslationService

from .trigger_reader import TriggerReader

logger = logging.getLogger(__name__)


@DatasetReader.register('bio_trigger')
class BIOTriggerReader(TriggerReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 trigger_label_namespace: str = 'event_labels',
                 multi_label: bool = True,
                 null_label: bool = False,
                 show_progress: bool = False,
                 translation_service: TranslationService = None,
                 **kwargs
                 ) -> None:
        self._show_progress = show_progress
        super().__init__(token_indexers,
                         trigger_label_namespace,
                         multi_label=multi_label,
                         null_label=null_label,
                         translation_service=translation_service,
                         **kwargs
                         )

    def _read_bio_sentences(self, file_path: str) -> Iterator[Tuple[List[str], List[str]]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens: List[str] = []
            bio_tags: List[str] = []
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    if tokens:
                        yield tokens, bio_tags
                    tokens = []
                    bio_tags = []
                    continue

                # Here we split the line only with the rightmost whitespace
                token, bio_tag = line.rsplit(maxsplit=1)
                tokens.append(token)
                bio_tags.append(bio_tag)

            if tokens:
                yield tokens, bio_tags

    def _decode_bio_spans(self, bio_tags: List[str]) -> Tuple[List[str], Dict[Tuple[int, int], str]]:
        labels: List[str] = []
        token_seqs: Dict[Tuple[int, int], str] = dict()

        # Offset of the current token of the span, inclusive
        first_idx = 0

        for idx, bio_tag in enumerate(bio_tags):
            bio, _, label = bio_tag.partition('-')
            label = label.replace(':', '.')  # Make label names TensorBoard-friendly

            if bio == 'O':
                labels.append('O')
                continue

            if bio == 'B':
                # Define the index of the first token of the span
                first_idx = idx
            else:
                # Remove previous span so the expanded one can be added
                del token_seqs[first_idx, idx - 1]

            token_seqs[first_idx, idx] = label
            labels.append(label)

        return labels, token_seqs

    def _build_instances(self, bio_sentences):
        for tokens, bio_tags in bio_sentences:
            tokens = [Token(token) for token in tokens]
            trigger_labels, trigger_token_seqs = self._decode_bio_spans(bio_tags)
            instance = self._build_instance(tokens,
                                            trigger_labels=trigger_labels,
                                            trigger_token_seqs=trigger_token_seqs)

            yield instance

    def _read_instances(self, file_path: str) -> Iterator[Instance]:
        logger.info(f'Reading {file_path}...')
        bio_sentences = self._read_bio_sentences(file_path)
        if self._show_progress:
            bio_sentences = tqdm(list(bio_sentences))

        instances = self._build_instances(bio_sentences)

        if self._show_progress:
            instances = list(instances)

        yield from instances
