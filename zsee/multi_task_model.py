from typing import Dict, Any, List

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models import Model


@Model.register("multi-task")
class MultiTaskModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 models: List[Model],
                 loss_weights: List[float] = None):
        super().__init__(vocab)

        self.models = torch.nn.ModuleList(models)
        if loss_weights is None:
            loss_weights = [1 for _ in self.models]
        self.loss_weights = loss_weights

    def forward(self, **kwargs) -> Dict[str, Any]:
        for model, loss_weight in reversed(list(zip(self.models, self.loss_weights))):
            try:
                output_dict = model(**kwargs)
                if output_dict is not None:
                    if 'loss' in output_dict:
                        output_dict['loss'] *= float(loss_weight)
                    return output_dict
            except TypeError as e:
                continue

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = dict()
        for model in self.models:
            metrics.update(model.get_metrics(reset))
        return metrics

    @classmethod
    def from_params(cls,
                    params: Params,
                    vocab: Vocabulary = None,
                    **extras):
        models = []
        for model_param in params.pop('models'):
            model = Model.from_params(params=model_param, vocab=vocab,
                                      models=models, **extras)
            models.append(model)

        return super().from_params(params, vocab=vocab,
                                   models=models, **extras)
