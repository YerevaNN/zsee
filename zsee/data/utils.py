from typing import List

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
