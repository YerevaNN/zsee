from typing import Dict, Iterator

import logging

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer, Field
from allennlp.data.fields import TextField

logger = logging.getLogger(__name__)


@DatasetReader.register('moses_parallel')
class MosesParallel(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer,
                 limit_samples: int = -1,
                 lazy: bool = False,
                 source_field: str = 'source_snt',
                 target_field: str = 'target_snt') -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer
        self._limit_samples = limit_samples
        self.source_field = source_field
        self.target_field = target_field

    def text_to_instance(self,
                         source_snt: str,
                         target_snt: str = None):
        fields: Dict[str, Field] = dict()

        source_snt_tokens = self._tokenizer.tokenize(source_snt)
        fields[self.source_field] = TextField(source_snt_tokens,
                                              self._token_indexers)

        if target_snt is None:
            return Instance(fields)

        target_snt_tokens = self._tokenizer.tokenize(target_snt)
        fields[self.target_field] = TextField(target_snt_tokens,
                                              self._token_indexers)
        return Instance(fields)

    def _read_lines(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.rstrip()

    def _read(self,
              source_path: str,
              target_path: str = None) -> Iterator[Instance]:
        if target_path is None:
            base, _, source_pair = source_path.rpartition('-')  # `News-Commentary.ar`, `-`, `en.ar`
            target_lang, _, source_lang = source_pair.partition('.')
            target_path = f'{base}-{target_lang}.{target_lang}'

        source_lines = self._read_lines(source_path)
        target_lines = self._read_lines(target_path)

        limit_samples = self._limit_samples

        for source, target in zip(source_lines, target_lines):
            yield self.text_to_instance(source, target)
            limit_samples -= 1

            if not limit_samples:
                break
