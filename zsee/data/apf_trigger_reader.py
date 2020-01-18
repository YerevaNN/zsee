import logging
import os
import re
from typing import Dict, Iterator, List, Tuple, Any

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.tokenizers import SentenceSplitter

from bs4 import BeautifulSoup
from pathlib import Path

from .trigger_reader import TriggerReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('ace2005_trigger')
@DatasetReader.register('apf_trigger')
class APFTriggerReader(TriggerReader):
    """
    Trigger reader which reads APF (Ace Program Format) files corresponding
    to the paths given in the source file.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer,
                 sentence_splitter: SentenceSplitter,
                 trigger_label_namespace: str = 'event_labels',
                 multi_label: bool = True,
                 null_label: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(token_indexers=token_indexers,
                         trigger_label_namespace=trigger_label_namespace,
                         multi_label=multi_label,
                         null_label=null_label,
                         lazy=lazy)
        self._tokenizer = tokenizer
        self._sentence_splitter = sentence_splitter

    def _read_doc_names(self, file_path: os.PathLike) -> Iterator[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove trailing newline character.
                # In case of empty line ignore it.
                name = line.rstrip('\n')
                if not name:
                    continue
                yield name

    def _read_sgm_text(self, sgm_path: Path) -> str:
        markup = sgm_path.read_text(encoding='utf-8')
        # soup = BeautifulSoup(markup, 'xml')
        pattern = re.compile(r'<.*?>',
                             re.MULTILINE | re.DOTALL)
        text = pattern.sub('', markup)
        text = text.replace('\n', ' ')
        return text

    def _process_sentence(self,
                          sentence: str,
                          sentence_offset: int = 0,
                          trigger_annotations: Dict[Tuple[int, int], str] = None) -> Instance:
        metadata: Dict[str, Any] = {
            "raw_sentence": sentence
        }

        # Sentence char-based boundaries
        sentence_start = sentence_offset
        sentence_end = sentence_start + len(sentence)

        # Event mention span tagging has to be done on token-level.
        tokens = self._tokenizer.tokenize(sentence)

        if trigger_annotations is None:
            return self._build_instance(tokens, **metadata)

        # Enumerate all the tokens and prepare char-to-token-id mappings
        # Store starts and ends separately to make multi-token mention mappings easier
        token_starts: Dict[int, int] = dict()
        token_ends: Dict[int, int] = dict()
        for idx, token in enumerate(tokens):
            start = token.idx
            end = start + len(token.text)
            token_starts[start] = idx
            token_ends[end] = idx

        # Prepare both char-based and token-based span offsets.
        trigger_labels = ['O' for token in tokens]
        trigger_char_seqs: Dict[Tuple[int, int], str] = dict()
        trigger_token_seqs: Dict[Tuple[int, int], str] = dict()
        for (start, end), label in trigger_annotations.items():
            # Filter only those non-empty charseq mentions
            # within the sentence boundaries
            if not sentence_start <= start < end < sentence_end:
                continue

            # Mention char-based start offset, relative to sentence, inclusive
            relative_start = start - sentence_start
            # Mention char-based end offset, relative to sentence, exclusive
            relative_end = end - sentence_start

            trigger_char_seqs[relative_start, relative_end] = label

            # Mention token-based start offset, relative to sentence, inclusive
            start_idx = token_starts.get(relative_start)
            # Mention token-based end offset, relative to sentence, inclusive
            end_idx = token_ends.get(relative_end)

            # In case of if exact mapping between char-based mention and
            # tokens isn't possible, just omit the event.
            # For evaluation phase, only `trigger_char_seqs` are used
            # for fair comparison without any ground truth information loss.
            if start_idx is None or end_idx is None:
                logger.warning(f'Mention-Token span mismatch:'
                               f'{sentence[relative_start:relative_end]!r} !='
                               f'...{tokens[start_idx or end_idx or 0]}...')
                continue

            # Now, populate `labels` and `trigger_token_seqs` with retained events.
            # These retained events should be only used for training.
            trigger_token_seqs[start_idx, end_idx] = label
            for idx in range(start_idx, end_idx + 1):
                trigger_labels[idx] = label

        return self._build_instance(tokens, trigger_labels,
                                    trigger_token_seqs=trigger_token_seqs,
                                    trigger_char_seqs=trigger_char_seqs,
                                    **metadata)

    def _process_doc(self,
                     text: str,
                     trigger_annotations: Dict[Tuple[int, int], str] = None) -> Iterator[Instance]:
        # Split the sentences. Non-destructive splitter is needed. e.g. `spacy_raw`
        sentences = self._sentence_splitter.split_sentences(text)
        assert ''.join(sentences) == text

        # Now, process all the (possibly annotated) sentences.
        sentence_offset = 0
        for idx, sentence in enumerate(sentences):
            # TODO skip first few lines / headers
            yield self._process_sentence(sentence,
                                         sentence_offset,
                                         trigger_annotations)
            sentence_offset += len(sentence)

    def _read_doc(self,
                  sgm_path: Path,
                  apf_xml_path: Path = None) -> Iterator[Instance]:
        text = self._read_sgm_text(sgm_path)

        trigger_annotations = None
        if apf_xml_path is not None:
            trigger_annotations = self._read_apf_xml_annotataions(apf_xml_path, text)

        yield from self._process_doc(text, trigger_annotations)

    def _read_apf_xml_annotataions(self,
                                   apf_path: Path,
                                   text: str = None) -> Dict[Tuple[int, int], str]:
        # Annotations are given on document-level, so we are going to capture
        # char-level event mention spans. Only then the mentions will be filtered
        # during sentence splitting.
        apf_xml = apf_path.read_text(encoding='utf-8')
        soup = BeautifulSoup(apf_xml, 'xml')

        trigger_annotations: Dict[Tuple[int, int], str] = dict()
        for event in soup.find_all('event'):
            event_type: str = event.get('TYPE')
            event_subtype: str = event.get('SUBTYPE')

            for event_mention in event.find_all('event_mention'):
                charseq = event_mention.anchor.charseq

                # Char-level start offset, inclusive, relative to document start.
                start = int(charseq.get('START'))
                # Char-level end offset, exclusive, relative to document start.
                end = int(charseq.get('END')) + 1
                # Type and Subtype are stored as a single string Type.Subtype
                label = f'{event_type}.{event_subtype}'

                mention = start, end
                trigger_annotations[mention] = label

                if text is not None:
                    expected = charseq.text.replace('\n', ' ')
                    text_found = text[start:end]
                    if text_found != expected:
                        logger.warning(f'Mention text mismatch: '
                                       f'Expected: {expected}, '
                                       f'Found: {text_found}')
                        # raise ValueError('Mention text mismatch')

        return trigger_annotations

    def _read(self, file_path: str) -> Iterator[Instance]:
        num_docs_read = 0
        num_instances_read = 0

        # We consider lines of file at `file_path` paths of the documents
        # to read. Paths are considered relative to the catalog.
        file_path = Path(file_path)
        search_dir = file_path.parent

        for doc_name in self._read_doc_names(file_path):
            # Text and annotations are given with `.sgm` and `.apf.xml` file pairs.
            sgm_path = search_dir / f'{doc_name}.sgm'
            apf_xml_path = search_dir / f'{doc_name}.apf.xml'
            # If the `.apf.xml` annotation file is not found,
            # process document without annotations. Useful for inference.
            if not apf_xml_path.is_file():
                apf_xml_path = None

            # In case of if sgm / xml parsing failed, just skip the document.
            # try:
            for instance in self._read_doc(sgm_path, apf_xml_path):
                yield instance
                num_instances_read += 1
            # except ValueError:
            #     logger.warning(f'Parse error. Ignoring a document {doc_name}')
            num_docs_read += 1

        logger.info(f'Read {num_docs_read} docs')
        logger.info(f'Read {num_instances_read} instances')
