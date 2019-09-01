from typing import Any, Dict, List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('trigger-tagger')
class TriggerTaggerPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True,
                                            keep_spacy_tokens=True)

    @overrides
    def load_line(self, line: str) -> JsonDict:
        return {
            "sentence": line
        }

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    def _process_outputs(self, outputs):
        outputs["hierplane_tree"] = self._build_hierplane_tree(outputs)
        return outputs

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs = self._process_outputs(outputs)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            output = self._process_outputs(output)
        return sanitize(outputs)


    def _build_hierplane_tree(self, output_dict: Dict[str, Any]) -> JsonDict:
        text = output_dict['raw_sentence']
        triggers = output_dict['pred_trigger_char_seqs']

        def build_spans(spans, range_start, range_end):
            spans = list(spans)
            spans.sort()

            new_spans = []

            cursor = range_start
            for (start, end), _ in spans:
                if cursor < start:
                    new_spans.append({
                        "start": cursor,
                        "end": start,
                        "spanType": "ignored"
                    })
                cursor = end

            if cursor < range_end:
                new_spans.append({
                    "start": cursor,
                    "end": range_end,
                    "spanType": "ignored"
                })

            return new_spans


        hierplane_tree = {
          "text": text,
          "root": {
            "nodeType": "top-level-and",
            "word": "and",
            "spans": build_spans(triggers, 0, len(text)),
            "children": [
              {
                "nodeType": "event",
                "word": text[start:end],
                "attributes": [label],
                "link": "none",
                "spans": [
                  {
                    "start": start,
                    "end": end
                  }
                ]
              } for (start, end), label in triggers
            ]
          }
        }

        if not hierplane_tree["root"]["children"]:
            del hierplane_tree["root"]["children"]

        return hierplane_tree