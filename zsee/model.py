from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder


@Model.register('zsee')
class ZSEE(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        pass

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().decode(output_dict)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return super().get_metrics(reset)
