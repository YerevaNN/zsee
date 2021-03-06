from collections import OrderedDict
from typing import Dict

import logging
import torch
from overrides import overrides
from torch import Tensor
from torch.nn import Module

from allennlp.common import Params, Lazy
from allennlp.models import load_archive
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.nn.util import get_text_field_mask

from .modules import BiasOnly, Normalization


logger = logging.getLogger(__name__)


@TextFieldEmbedder.register('pretrained')
class PretrainedModelTextFieldEmbedder(TextFieldEmbedder):

    def __init__(self,
                 archive_file: str,
                 weights_file: str = None,
                 frozen: bool = False,
                 key: str = '_text_field_embedder'):
        super().__init__()
        self._frozen = frozen

        archive = load_archive(archive_file=archive_file,
                               weights_file=weights_file)

        pretrained_model = archive.model
        self._text_field_embedder = getattr(pretrained_model, key)

        if self._frozen is True:
            for param in self._text_field_embedder.parameters():
                param.requires_grad = False

    def forward(self,
                text_field_input: Dict[str, torch.Tensor],
                *args, **kwargs) -> torch.Tensor:
        return self._text_field_embedder.forward(text_field_input=text_field_input,
                                                 *args, **kwargs)

    def get_output_dim(self) -> int:
        return self._text_field_embedder.get_output_dim()

    @overrides
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if self._frozen:
            return OrderedDict()
        state_dict = super().state_dict(destination, prefix, keep_vars)
        return state_dict

    @overrides
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        # Here we build state dict with current values for all the keys,
        # both the tracked and not tracked modules.
        # Note, this calls ``state_dict()`` of super class, because we already
        # had overridden it to return only values for tracked modules.
        current_state = super().state_dict(prefix=prefix)

        keys_to_remove = []
        for key in state_dict:
            if key.startswith(prefix):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]

        for key in current_state:
            state_dict[key] = current_state[key]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    #     # Here we build state dict with current values for all the keys,
    #     # both the tracked and not tracked modules.
    #     # Note, this calls ``state_dict()`` of super class, because we already
    #     # had overridden it to return only values for tracked modules.
    #     current_state = super().state_dict(prefix=prefix)
    #
    #     # Provided ``state_dict`` contains only target values of tracked modules.
    #     # We alter so now it will include current values of untracked modules as well.
    #     for key in current_state:
    #         if key in state_dict:
    #             continue
    #         state_dict[key] = current_state[key]
    #
    #     return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
    #                                          missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_params(cls, params: Params, **extras) -> TextFieldEmbedder:
        archive_file = params.pop('archive_file')
        weights_file = params.pop('weights_file', None)
        frozen = params.pop('frozen', False)
        key = params.pop('key', default='_text_field_embedder')
        return cls(archive_file, weights_file, frozen, key)


@TextFieldEmbedder.register('mapped', 'from_partial_objects')
class MappedTextFieldEmbedder(TextFieldEmbedder):

    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 # frozen_embeddings: bool = True,
                 mapper: FeedForward = None,
                 bias: bool = True,
                 pre_normalization: Normalization = None,
                 post_normalization: Normalization = None,
                 normalization: Normalization = None):
        super().__init__()

        self._bias = bias
        self._text_field_embedder = text_field_embedder
        self._output_dim = self._text_field_embedder.get_output_dim()

        if normalization is not None:
            pre_normalization = normalization
        self._pre_normalization = pre_normalization
        self._post_normalization = post_normalization

        # self._frozen_embeddings = frozen_embeddings
        # if self._frozen_embeddings:
        #     self._text_field_embedder.requires_grad_(False)

        # TODO Make sure mapper supports time-distributed out-of-the-box.
        if mapper is not None:
            self._mapper = mapper
            self._output_dim = mapper.get_output_dim()
        else:
            if bias:
                self._mapper = BiasOnly(self._output_dim)
            else:
                self._mapper = Module()

    @classmethod
    def from_partial_objects(
            cls,
            text_field_embedder: TextFieldEmbedder,
            mapper: Lazy[FeedForward] = None,
            bias: bool = True,
            pre_normalization: Normalization = None,
            post_normalization: Normalization = None,
            normalization: Normalization = None
    ):
        text_field_embedder_dim = text_field_embedder.get_output_dim()
        logger.info(f"Original text field embedder dim: {text_field_embedder_dim}")
        mapper_ = mapper.construct(input_dim=text_field_embedder_dim)

        return cls(text_field_embedder=text_field_embedder,
                   mapper=mapper_,
                   bias=bias,
                   pre_normalization=pre_normalization,
                   post_normalization=post_normalization,
                   normalization=normalization)

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self,
                text_field_input: Dict[str, Dict[str, Tensor]],
                mapped: bool = True,
                both: bool = False,
                **kwargs) -> Tensor:

        # if self._frozen_embeddings:
        #     self._text_field_embedder.eval()

        # Shape: (batch_size, num_tokens, embedding_dim)
        embeddings = self._text_field_embedder.forward(text_field_input, **kwargs)

        mask = get_text_field_mask(text_field_input)
        if mask.size(1) != embeddings.size(1):
            mask = text_field_input['pretrained_transformer']['wordpiece_mask'].long()
        mask = mask.unsqueeze(-1).float()

        # TODO find mask, make the code generic and shared

        if self._pre_normalization:
            embeddings = self._pre_normalization(embeddings, mask=mask)
        # if self._pre_normalization is None:
        if not mapped:
            return embeddings

        # Shape: (batch_size, num_tokens, embedding_dim)
        mapped_embeddings = self._mapper(embeddings)

        if self._post_normalization:
            mapped_embeddings = self._post_normalization(mapped_embeddings, mask=mask)
        # if self._post_normalization is None:
        if not both:
            return mapped_embeddings

        return embeddings, mapped_embeddings
