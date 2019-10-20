import logging
import re

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
