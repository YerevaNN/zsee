from typing import List

from overrides import overrides

from allennlp.data import Tokenizer, Token
from nltk.parse.corenlp import CoreNLPParser


@Tokenizer.register('corenlp')
@Tokenizer.register('corenlp_remote')  # Compatibility with allennlp_corenlp_wordsplitter
class CoreNLPTokenizer(Tokenizer):
    def __init__(self,
                 url: str = 'http://localhost:9000',
                 encoding: str = 'utf-8',
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 ):
        self._parser = CoreNLPParser(url, encoding, 'pos')

        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:

        tokens = [Token(t) for t in self._parser.tokenize(text)]

        for start_token in self._start_tokens:
            if isinstance(start_token, int):
                token = Token(text_id=start_token, idx=0)
            else:
                token = Token(text=start_token, idx=0)
            tokens.insert(0, token)

        for end_token in self._end_tokens:
            if isinstance(end_token, int):
                token = Token(text_id=end_token, idx=0)
            else:
                token = Token(text=end_token, idx=0)
            tokens.append(token)

        return tokens
