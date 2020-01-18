from typing import List, Dict

import torch

from allennlp.data import TokenIndexer, Token, Vocabulary
from allennlp.data.tokenizers import SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


@SentenceSplitter.register('spacy_raw')
class SpacyRawSentenceSplitter(SpacySentenceSplitter):
    """
    This differs from the original ``SpacySentenceSplitter`` being
    non-destructive, i.e., concatenation of the sentences is the source text.
    """
    def split_sentences(self, text: str) -> List[str]:
        return [sent.string for sent in self.spacy(text).sents]

    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        return [[sentence.string for sentence in doc.sents] for doc in self.spacy.pipe(texts)]


@TokenIndexer.register('hash')
class TextHash(TokenIndexer):

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    def tokens_to_indices(
            self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        immutable_tokens = tuple(tokens)
        return {
            "hash": [hash(immutable_tokens)]
        }

    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        return {}

    def as_padded_tensor(
            self,
            tokens: Dict[str, List[int]],
            desired_num_tokens: Dict[str, int],
            padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(val, dtype=torch.long)
            for key, val in tokens.items()
        }

    def get_keys(self, index_name: str) -> List[str]:
        return ["hash"]
