from pathlib import Path
from .apf_trigger_reader import APFTriggerReader
from .utils import SpacyRawSentenceSplitter

if __name__ == "__main__":

    reader = APFTriggerReader(dict(), None, SpacyRawSentenceSplitter())

    file_path = 'data/LDC2006T06//en_train.files'

    file_path = Path(file_path)
    search_dir = file_path.parent

    snts = []
    for doc_name in reader._read_doc_names(file_path):
        # Text and annotations are given with `.sgm` and `.apf.xml` file pairs.
        sgm_path = search_dir / f'{doc_name}.sgm'
        sgm_text = reader._read_sgm_text(sgm_path)
        for idx, sentence in enumerate(reader._sentence_splitter.split_sentences(sgm_text)):
            sentence = sentence.strip()
            if idx < 2 or not sentence or len(sentence) < 10:
                continue
            snts.append(sentence)
        snts.append('')

    with open('en_train.corpus.txt', 'w', encoding='utf-8') as f:
        for snt in snts:
            f.write(snt)
            f.write('\n')
