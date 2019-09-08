from typing import List, Dict, Tuple, Iterator

from allennlp.data import Token, Instance, TokenIndexer, DatasetReader

from .trigger_reader import TriggerReader


@DatasetReader.register('bio_trigger')
class BIOTriggerReader(TriggerReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 trigger_label_namespace: str = 'event_labels',
                 lazy: bool = False) -> None:
        super().__init__(token_indexers,
                         trigger_label_namespace,
                         lazy)

    def _read_bio_sentences(self, file_path: str) -> Iterator[Tuple[List[str], List[str]]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens: List[str] = []
            bio_tags: List[str] = []
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    if tokens:
                        yield tokens, bio_tags
                    tokens.clear()
                    bio_tags.clear()
                    continue

                token, bio_tag = line.split('\t')
                tokens.append(token)
                bio_tags.append(bio_tag)

            if tokens:
                yield tokens, bio_tags

    def _decode_bio_spans(self, bio_tags: List[str]) -> Tuple[List[str], Dict[Tuple[int, int], str]]:
        labels: List[str] = []
        token_seqs: Dict[Tuple[int, int], str] = dict()

        # Offset of the current token of the span, inclusive
        first_idx = 0
        last_idx = 0

        for idx, bio_tag in enumerate(bio_tags):
            bio, _, label = bio_tag.partition('-')
            label = label.replace(':', '.')  # Make label names TensorBoard-friendly
            if bio == 'B':
                first_idx = idx
                last_idx = idx
                token_seqs[first_idx, last_idx] = label
            elif bio == 'I':
                del token_seqs[first_idx, last_idx]
                last_idx = idx
                token_seqs[first_idx, last_idx] = label
            else:
                label = 'O'

            labels.append(label)

        return labels, token_seqs


    def _read(self, file_path: str) -> Iterator[Instance]:
        for tokens, bio_tags in self._read_bio_sentences(file_path):
            tokens = [Token(token) for token in tokens]
            trigger_labels, trigger_token_seqs = self._decode_bio_spans(bio_tags)
            yield self._build_instance(tokens,
                                       trigger_labels=trigger_labels,
                                       trigger_token_seqs=trigger_token_seqs)
