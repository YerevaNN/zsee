import logging
import os
import re
from collections import defaultdict

from typing import List, Optional, Union, Dict

import torch

from allennlp.common import Params
from allennlp.common.checks import parse_cuda_device
from allennlp.common.util import get_frozen_and_tunable_parameter_names
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training import CallbackTrainer, TrainerBase
from allennlp.training.callbacks import Callback
from allennlp.training.optimizers import Optimizer

from .data.configuration import DataConfigurationBase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register('multi-task')
class MultiTaskTrainer(CallbackTrainer):

    def __init__(self,
                 model: Model,
                 data_config: DataConfigurationBase,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int = 20,
                 shuffle: bool = True,
                 serialization_dir: Optional[str] = None,
                 cuda_device: Union[int, List] = -1,
                 callbacks: List[Callback] = None) -> None:
        # We want to reuse constructor logic behind the original CallbackTrainer.
        # However, it expects single iterable as training data.
        # We'll pass an empty iterable, then will overwrite it later.

        train_instances = data_config.instances['_train']
        train_iterator = data_config.iterators['_train']
        super().__init__(model, train_instances, train_iterator,
                         optimizer, num_epochs, shuffle,
                         serialization_dir, cuda_device,
                         callbacks)
        self.data_config = data_config
        self.datasets_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

    @classmethod
    def trainer_pieces(cls,
                       params: Params,
                       serialization_dir: str,
                       recover: bool = False,
                       cache_directory: str = None,
                       cache_prefix: str = None):
        data_params = params.pop('data')
        data_config = DataConfigurationBase.from_params(data_params,
                                                        cache_directory=cache_directory,
                                                        cache_prefix=cache_prefix)

        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
            params.pop("vocabulary", {})
        else:
            vocab = Vocabulary.from_params(
                params.pop("vocabulary", {}),
                # Using a generator comprehension here is important
                # because, being lazy, it allows us to not iterate over the
                # dataset when directory_path is specified.
                (instance for key, dataset in data_config.instances.items()
                 if key in data_config.datasets_for_vocab_creation for instance in dataset)
            )

        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        # If vocab extension is ON for training, embedding extension should also be
        # done. If vocab and embeddings are already in sync, it would be a no-op.
        model.extend_embedder_vocab()

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        data_config.index_with(model.vocab)

        no_grad_regexes = params.get("trainer").pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = \
            get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        return data_config, model

    @classmethod
    def from_params(cls,
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None):

        data_config, model = cls.trainer_pieces(params, serialization_dir, recover, cache_directory, cache_prefix)  # pylint: disable=no-member
        params = params.pop('trainer')

        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        callbacks_params = params.pop("callbacks", [])
        callbacks: List[Callback] = [Callback.from_params(params=callback_params,
                                                          model=model,
                                                          optimizer=optimizer,
                                                          # instances=pieces['train_dataset'],
                                                          # iterator=pieces['iterator'],
                                                          shuffle=shuffle,
                                                          # validation_data=data_config.instances['_validation'],
                                                          validation_iterator=data_config.iterators['_validation'],
                                                          serialization_dir=serialization_dir,
                                                          other_data=data_config.instances
                                                          )
                                     for callback_params in callbacks_params]

        params.assert_empty(cls.__name__)
        return cls(model,
                   data_config,
                   optimizer,
                   num_epochs=num_epochs,
                   shuffle=shuffle,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   callbacks=callbacks)
