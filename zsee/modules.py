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
