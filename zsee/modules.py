import math

from torch import Tensor
from torch.nn import Module, Parameter, init

from allennlp.common import FromParams


class BiasOnly(Module, FromParams):
    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim
        self._bias = Parameter(Tensor(input_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self._bias)

    def forward(self, inputs: Tensor):
        return inputs + self._bias
