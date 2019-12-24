from typing import Any, Dict, List

from overrides import overrides

import numpy as np

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('trigger-tagger')
class TriggerTaggerPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 language: str = 'en_core_web_sm',
                 sanitize: bool = True) -> None:
        super().__init__(model, dataset_reader)
        self._sanitize = sanitize
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True,
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
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    def _decode_wrt_mask(self, tensor, mask):
        return np.stack([
            vector
            for vector, not_masked
            in zip(tensor, mask)
            if not_masked
        ])

    def _process_outputs(self, output_dict):
        # TODO implement BIO-based NER visualization

        # Construct hierplace tree
        hierplane_tree = self._build_hierplane_tree(output_dict)
        if hierplane_tree:
            output_dict["hierplane_tree"] = hierplane_tree

        # Unmask and decode embeddings w.r.t. mask
        mask = output_dict["mask"]

        contextual_embeddings = output_dict["contextual_embeddings"]
        output_dict["contextual_embeddings"] = self._decode_wrt_mask(contextual_embeddings,
                                                                     mask)
        encoder_embeddings = output_dict["encoder_embeddings"]
        output_dict["encoder_embeddings"] = self._decode_wrt_mask(encoder_embeddings,
                                                                  mask)

        return output_dict

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs = self._process_outputs(outputs)

        if self._sanitize:
            outputs = sanitize(outputs)
        return outputs

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            output = self._process_outputs(output)

        if self._sanitize:
            outputs = sanitize(outputs)
        return outputs

    def _build_hierplane_tree(self, output_dict: Dict[str, Any]) -> JsonDict:
        text = output_dict.get('raw_sentence')
        triggers = output_dict.get('pred_trigger_char_seqs')

        if text is None or triggers is None:
            return dict()

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
