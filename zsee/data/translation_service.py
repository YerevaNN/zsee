import logging
from collections import defaultdict
from pathlib import Path
from typing import overload, List
from typing import Dict

from overrides import overrides

from allennlp.common import Registrable
from allennlp.data import Token, Tokenizer

logger = logging.getLogger(__name__)


class TranslationService(Registrable):

    @overload
    def __call__(self, source: str) -> str:
        ...

    @overload
    def __call__(self, source: List[Token]) -> List[Token]:
        ...

    def __call__(self, source):
        if isinstance(source, str):
            return self._translate(source)
        return self._translate_tokens(source)

    def _translate(self, source_snt: str) -> str:
        raise NotImplementedError

    def _translate_tokens(self, source_tokens: List[Token]) -> List[Token]:
        raise NotImplementedError

    def close(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@TranslationService.register('cached')
class CachedTranslationService(TranslationService):
    def __init__(self,
                 tokenizer: Tokenizer,
                 source_lang: str = 'src',
                 target_lang: str = 'tgt',
                 cache_dir: str = 'data/mt'):
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.cache_dir = Path(cache_dir)
        direction = f'{source_lang}-{target_lang}'
        self.source_path = self.cache_dir / f'{direction}.{source_lang}'
        self.target_path = self.cache_dir / f'{direction}.{target_lang}'
        self.tokenizer = tokenizer

        self.translations: Dict[str, str] = defaultdict(str)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_file = self.source_path.open('a+', encoding='utf-8')
        self.source_file.seek(0)
        self.target_file = self.target_path.open('a+', encoding='utf-8')
        self.target_file.seek(0)

        for source_snt, target_snt in zip(self.source_file, self.target_file):
            source_snt: str = source_snt.strip('\n')
            target_snt: str = target_snt.strip('\n')

            if not source_snt and not target_snt:
                continue

            self.translations[source_snt] = target_snt

        # If we intend to translate with some API:
        #    self.translator = Translator()

    @overrides
    def close(self):
        self.translations = None
        if self.source_file is not None:
            self.source_file.close()
            self.source_file = None
        if self.target_file is not None:
            self.target_file.close()
            self.target_file = None

    def _report_missing(self,
                        source_snt: str,
                        target_snt: str = ''):
        self.source_file.write(source_snt)
        self.source_file.write('\n')
        self.target_file.write(target_snt)
        self.target_file.write('\n')

    @overrides
    def _translate(self, source_snt: str) -> str:
        target_snt = self.translations[source_snt]

        if not target_snt:
            # try:
            #     translation = self.translator.translate(source_snt,
            #                                             src=self.source_lang,
            #                                             dest=self.target_lang)
            #     target_snt = translation.text
            # except Exception as e:
            #     target_snt = 'FAILED_TO_TRANSLATE'
            #     logger.error(e)
            self._report_missing(source_snt, target_snt)

        return target_snt

    def _tokenize(self, sentence: str) -> List[Token]:
        return self.tokenizer.tokenize(sentence)

    def _detokenize(self, tokens: List[Token]) -> str:
        tokens_with_whitespaces = [
            getattr(token, 'text_with_ws', f'{token} ')  # use non-destructive tokens
            for token in tokens                          # if available
        ]
        return ''.join(tokens_with_whitespaces).rstrip(' ')

    @overrides
    def _translate_tokens(self, source_tokens: List[Token]) -> List[Token]:
        source_snt = self._detokenize(source_tokens)
        target_snt = self._translate(source_snt)
        target_tokens = self._tokenize(target_snt)
        return target_tokens
