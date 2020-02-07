import logging
import re
from typing import List

from torch import Tensor

from allennlp.training.callbacks import Callback, handle_event, Events

logger = logging.getLogger(__name__)


@Callback.register("orthonormalize")
class OrthonormalizeCallback(Callback):

    def __init__(self,
                 regex: str,
                 beta: float = 0.01) -> None:
        self.regex = regex
        self.beta = beta

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer):
        for name, parameter in trainer.model.named_parameters():
            if not re.search(self.regex, name):
                continue

            W: Tensor = parameter
            W_hat = W @ W.t() @ W

            W.data *= (1 + self.beta)
            W.data -= self.beta * W_hat


@Callback.register("log_singular_values")
class LogSingularValues(Callback):

    def __init__(self,
                 regex: str) -> None:
        self.regex = regex

    @handle_event(Events.EPOCH_START)
    def on_epoch_start(self, trainer):
        for name, parameter in trainer.model.named_parameters():
            if not re.search(self.regex, name):
                continue

            w: Tensor = parameter
            u, s, v = w.svd(compute_uv=False)

            singular_values = s.detach().cpu().numpy()
            logger.info(f'Singular Values of {name}:')
            logger.info(f'{singular_values[0]:.2f}, {singular_values[1]:.2f}, {singular_values[2]:.2f} ... {singular_values[-2]:.2f}, {singular_values[-1]:.2f}')
            logger.info(f'min={singular_values.min():.2f}, '
                        f'mean={singular_values.mean():.2f}, '
                        f'max={singular_values.max():.2f}')
