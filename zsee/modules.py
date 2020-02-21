import logging
from abc import ABC
from typing import Type, Union, Any, Tuple

from overrides import overrides

from torch import Tensor
from torch.nn import Module, Parameter, init, Linear
from torch.nn.modules.loss import _WeightedLoss, _Loss

from allennlp.common import FromParams
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn.util import masked_softmax

logger = logging.getLogger(__name__)


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


class ClassBalancedFocalLoss(Module):

    def __init__(self,
                 cls: Union[Type[_WeightedLoss], Type[_Loss]],
                 *,
                 gamma: float = 0,
                 beta: float = None,
                 weight: Tensor = None,
                 class_statistics: Tensor = None,
                 reduction: str = 'mean'):
        super().__init__()

        if weight is None and class_statistics is not None:
            logger.info('Calculating class weights from class statistics.')
            logger.info(f'Inverse-rank weighting will result in {1 / class_statistics}')
            if beta is None or beta == 1:
                beta = 1
                weight: Tensor = 1 / class_statistics
                logger.info('Inverse-rank weighting is applied.')
            else:
                weight: Tensor = (1 - beta) / (1 - beta ** class_statistics)
                logger.info('Weights are calculated w.r.t. effective number of samples.')
                logger.info(f'{weight}')

            weight /= weight.mean()

        self.loss = cls(weight=weight, reduction='none')
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Shape: (batch_size, num_classes)
        loss = self.loss.forward(y_pred, y_true)

        if self.gamma != 0:
            loss *= (1 - y_pred).pow(self.gamma)  # TODO fix

        if self.reduction == 'mean':
            # Shape: ()
            loss = loss.mean()
        elif self.reduction == 'sum':
            # Shape: ()
            loss = loss.sum()
        else:
            raise NotImplementedError

        return loss


@Seq2VecEncoder.register('attention')
class MultiHeadAttentionPooler(Seq2VecEncoder):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 projection: bool = True
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.f = Linear(input_dim, output_dim)
        if projection:
            self.g = Linear(input_dim, 1)
        else:
            assert output_dim == 1
            self.g = None

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
            self,
            value: Tensor,
            mask: Tensor) -> Tensor:
        # Shape: (batch_size, num_tokens, input_dim)
        # ->
        # Either
        # Shape: (batch_size, input_dim)
        # Or
        # Shape: (batch_size, output_dim)

        batch_size, num_tokens, input_dim = value.size()

        # Shape: (batch_size, num_tokens, num_heads)
        a = self.f(value)
        # Shape: (batch_size, num_heads, num_tokens)
        s = masked_softmax(a, mask.unsqueeze(-1), dim=1).transpose(-1, -2)

        # Shape: (batch_size, num_heads, input_dim)
        weighted_sum = s.bmm(value)

        if self.g is None:
            # Shape: (batch_size, input_dim)
            return weighted_sum.squeeze(dim=1)

        # Shape: (batch_size, num_heads, 1)
        g = self.g(weighted_sum)

        # Shape: (batch_size, num_heads)
        return g.squeeze(dim=-1)


class Normalization(Seq2SeqEncoder, ABC):
    default_implementation = 'default'

    def __init__(self,
                 input_dim: int = None):
        super().__init__()
        self._input_dim = input_dim

    def get_input_dim(self) -> int:
        assert self._input_dim is not None
        return self._input_dim

    def get_output_dim(self) -> int:
        assert self._input_dim is not None
        return self._input_dim


@Normalization.register('mean')
class MeanNormalization(Normalization):
    def __init__(self,
                 dims: Tuple[int] = (0, 1),
                 **kwargs):
        super().__init__(**kwargs)
        self._dims = dims

    def forward(self,
                tensor: Tensor,
                mask: Tensor) -> Tensor:
        # TODO Optional mask
        clean_tensor = tensor * mask
        total = clean_tensor.sum(dim=self._dims, keepdim=True)
        num_elements = mask.sum(dim=self._dims, keepdim=True)
        mean = total / num_elements
        masked_centered = (clean_tensor - mean) * mask
        return masked_centered


@Normalization.register('default')
@Normalization.register('mean-std')
class MeanStdNormalization(Normalization):
    def __init__(self,
                 dims: Tuple[int] = (0, 1),
                 eps: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self._dims = dims
        self._eps = eps

    def forward(self,
                tensor: Tensor,
                mask: Tensor) -> Tensor:
        # TODO Optional mask

        # Shape: (batch_size, seq_length, num_channels)

        # Shape: (batch_size, seq_length, num_channels)
        clean_tensor = tensor * mask
        # Shape: (1, 1, num_channels)
        total = clean_tensor.sum(dim=self._dims, keepdim=True)
        # Shape: (1, 1, num_channels)
        num_elements = mask.sum(dim=self._dims, keepdim=True)
        # Shape: (1, 1, num_channels)
        mean = total / num_elements
        # Shape: (batch_size, seq_length, num_channels)
        masked_centered = (clean_tensor - mean) * mask
        # return masked_centered
        # std = torch.sqrt((masked_centered * masked_centered).sum() / num_elements + self.eps)
        # return self.gamma * (tensor - mean) / (std + self.eps) + self.beta

        # Shape: (1, 1, num_channels)
        var = (masked_centered * masked_centered).sum(dim=self._dims) / num_elements
        std = var.sqrt()

        return masked_centered / (std + self._eps)
