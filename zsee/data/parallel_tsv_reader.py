from typing import Dict, Iterator

import logging

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer, Field
from allennlp.data.fields import TextField

from pathlib import Path

logger = logging.getLogger(__name__)


@DatasetReader.register('parallel_tsv')
class ParallelTSVReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer

    def _build_instance(self,
                        source_snt: str,
                        target_snt: str = None):
        fields: Dict[str, Field] = dict()

        source_snt_tokens = self._tokenizer.tokenize(source_snt)
        fields['source_snt'] = TextField(source_snt_tokens,
                                         self._token_indexers)

        if target_snt is None:
            return Instance(fields)

        target_snt_tokens = self._tokenizer.tokenize(target_snt)
        fields['target_snt'] = TextField(target_snt_tokens,
                                         self._token_indexers)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        file_path = Path(file_path)
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                yield self.text_to_instance(line)

    def text_to_instance(self,
                         source_snt: str,
                         target_snt: str = None) -> Instance:
        left, _, right = source_snt.partition('\t')
        yield self._build_instance(left,
                                   target_snt or right or None)
