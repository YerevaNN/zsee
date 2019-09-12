from typing import List
from overrides import overrides

from allennlp.modules import TokenEmbedder
from allennlp.modules.token_embedders import BertEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel


@TokenEmbedder.register("bert-pretrained-only")
class PretrainedBertOnlyEmbedder(BertEmbedder):
    def __init__(self,
                 pretrained_model: str,
                 requires_grad: bool = False,
                 top_layer_only: bool = False,
                 untracked_keys: List[str] = None) -> None:
        model = PretrainedBertModel.load(pretrained_model)

        for param in model.parameters():
            param.requires_grad = requires_grad

        if untracked_keys is None:
            untracked_keys = ['bert_model']

        self._untracked_keys = untracked_keys
        super().__init__(bert_model=model, top_layer_only=top_layer_only)

    @overrides
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super().state_dict(destination, prefix, keep_vars)

        keys_to_remove = []
        for key in self._untracked_keys:
            for prefixed_key in destination:
                if prefixed_key.startswith(prefix + key + '.'):
                    keys_to_remove.append(prefixed_key)

        for key in keys_to_remove:
            del destination[key]

        return destination

    @overrides
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Here we build state dict with current values for all the keys,
        # both the tracked and not tracked modules.
        # Note, this calls ``state_dict()`` of super class, because we already
        # had overridden it to return only values for tracked modules.
        current_state = super().state_dict(prefix=prefix)

        # Provided ``state_dict`` contains only target values of tracked modules.
        # We alter so now it will include current values of untracked modules as well.
        for key in current_state:
            if key in state_dict:
                continue
            state_dict[key] = current_state[key]

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)
