import math

from overrides import overrides
from torch import Tensor
from torch.nn import Module, Parameter, init

from allennlp.common import FromParams
from allennlp.modules import Seq2VecEncoder


class BiasOnly(Module, FromParams):
    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim
        self._bias = Parameter(Tensor(input_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self._bias)

    @overrides
    def forward(self, inputs: Tensor):
        return inputs + self._bias


@Seq2VecEncoder.register("first_token")
class FirstTokenPooler(Seq2VecEncoder):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self._embedding_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self,
                tokens: Tensor,
                mask: Tensor = None):
        # Collect embeddings of first tokens, presumably [CLS] or similar
        return tokens[:, 0]
