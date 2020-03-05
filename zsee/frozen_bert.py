import re
from typing import Dict, List, Union
import logging

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util

logger = logging.getLogger(__name__)


@TokenEmbedder.register('frozen-bert')
class FrozenBertEmbedder(TokenEmbedder):
    
    def __init__(
            self,
            pretrained_model: str,
            requires_grad: Union[bool, str] = False,
            top_layer_only: bool = False,
            max_pieces: int = 512,
            num_start_tokens: int = 1,
            num_end_tokens: int = 1,
            scalar_mix_parameters: List[float] = None,
            untracked_keys: List[str] = None,
            always_eval_mode: bool = False
    ) -> None:

        super().__init__()

        if untracked_keys is None:
            if not requires_grad:
                untracked_keys = ['bert_model']
            else:
                untracked_keys = []

        self.untracked_keys = untracked_keys
        self.always_eval_mode = always_eval_mode

        bert_model = PretrainedBertModel.load(pretrained_model)

        for name, param in bert_model.named_parameters():
            if isinstance(requires_grad, bool):
                param.requires_grad = requires_grad
            elif isinstance(requires_grad, str):
                matches = re.search(requires_grad, name) is not None
                param.requires_grad = matches
            else:
                raise NotImplementedError

        self.bert_model = bert_model
        self.output_dim = bert_model.config.hidden_size
        self.max_pieces = max_pieces
        self.num_start_tokens = num_start_tokens
        self.num_end_tokens = num_end_tokens

        if not top_layer_only:
            self._scalar_mix = ScalarMix(
                bert_model.config.num_hidden_layers + 1,
                do_layer_norm=False,
                initial_scalar_parameters=scalar_mix_parameters,
                trainable=scalar_mix_parameters is None,
                )
        else:
            self._scalar_mix = None

        bert_model.encoder.output_hidden_states = True

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
            self,
            input_ids: torch.LongTensor,
            token_type_ids: torch.LongTensor = None,
            **kwargs
    ) -> torch.Tensor:
        """
        # Parameters

        input_ids : `torch.LongTensor`
            The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
        token_type_ids : `torch.LongTensor`, optional
            If an input consists of two sentences (as in the BERT paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            (the default BertIndexer doesn't) then it's assumed to be all 0s.
        """

        batch_size, full_seq_len = input_ids.size(0), input_ids.size(-1)
        initial_dims = list(input_ids.shape[:-1])

        # The embedder may receive an input tensor that has a sequence length longer than can
        # be fit. In that case, we should expect the wordpiece indexer to create padded windows
        # of length `self.max_pieces` for us, and have them concatenated into one long sequence.
        # E.g., "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ..."
        # We can then split the sequence into sub-sequences of that length, and concatenate them
        # along the batch dimension so we effectively have one huge batch of partial sentences.
        # This can then be fed into BERT without any sentence length issues. Keep in mind
        # that the memory consumption can dramatically increase for large batches with extremely
        # long sentences.
        needs_split = full_seq_len > self.max_pieces
        last_window_size = 0
        if needs_split:
            # Split the flattened list by the window size, `max_pieces`
            split_input_ids = list(input_ids.split(self.max_pieces, dim=-1))

            # We want all sequences to be the same length, so pad the last sequence
            last_window_size = split_input_ids[-1].size(-1)
            padding_amount = self.max_pieces - last_window_size
            split_input_ids[-1] = F.pad(split_input_ids[-1], pad=[0, padding_amount], value=0)

            # Now combine the sequences along the batch dimension
            input_ids = torch.cat(split_input_ids, dim=0)

            if token_type_ids is not None:
                # Same for token_type_ids
                split_token_type_ids = list(token_type_ids.split(self.max_pieces, dim=-1))

                last_window_size = split_token_type_ids[-1].size(-1)
                padding_amount = self.max_pieces - last_window_size
                split_token_type_ids[-1] = F.pad(
                    split_token_type_ids[-1], pad=[0, padding_amount], value=0
                )

                token_type_ids = torch.cat(split_token_type_ids, dim=0)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        # TODO always eval mode?
        if self.always_eval_mode:
            raise NotImplementedError

        # with torch.no_grad():
        # input_ids may have extra dimensions, so we reshape down to 2-d
        # before calling the BERT model and then reshape back at the end.

        # TODO what about pooling
        _, _, all_encoder_layers = self.bert_model(
            input_ids=util.combine_initial_dims(input_ids),
            token_type_ids=util.combine_initial_dims(token_type_ids),
            attention_mask=util.combine_initial_dims(input_mask),
        )

        all_encoder_layers = torch.stack(all_encoder_layers)

        if needs_split:
            # First, unpack the output embeddings into one long sequence again
            unpacked_embeddings = torch.split(all_encoder_layers, batch_size, dim=1)
            unpacked_embeddings = torch.cat(unpacked_embeddings, dim=2)

            # Next, select indices of the sequence such that it will result in embeddings representing the original
            # sentence. To capture maximal context, the indices will be the middle part of each embedded window
            # sub-sequence (plus any leftover start and final edge windows), e.g.,
            #  0     1 2    3  4   5    6    7     8     9   10   11   12    13 14  15
            # "[CLS] I went to the very fine [SEP] [CLS] the very fine store to eat [SEP]"
            # with max_pieces = 8 should produce max context indices [2, 3, 4, 10, 11, 12] with additional start
            # and final windows with indices [0, 1] and [14, 15] respectively.

            # Find the stride as half the max pieces, ignoring the special start and end tokens
            # Calculate an offset to extract the centermost embeddings of each window
            stride = (self.max_pieces - self.num_start_tokens - self.num_end_tokens) // 2
            stride_offset = stride // 2 + self.num_start_tokens

            first_window = list(range(stride_offset))

            max_context_windows = [
                i
                for i in range(full_seq_len)
                if stride_offset - 1 < i % self.max_pieces < stride_offset + stride
            ]

            # Lookback what's left, unless it's the whole self.max_pieces window
            if full_seq_len % self.max_pieces == 0:
                lookback = self.max_pieces
            else:
                lookback = full_seq_len % self.max_pieces

            final_window_start = full_seq_len - lookback + stride_offset + stride
            final_window = list(range(final_window_start, full_seq_len))

            select_indices = first_window + max_context_windows + final_window

            initial_dims.append(len(select_indices))

            recombined_embeddings = unpacked_embeddings[:, :, select_indices]
        else:
            recombined_embeddings = all_encoder_layers

        # Recombine the outputs of all layers
        # (layers, batch_size * d1 * ... * dn, sequence_length, embedding_dim)
        # recombined = torch.cat(combined, dim=2)
        input_mask = (recombined_embeddings != 0).long()

        if self._scalar_mix is not None:
            mix = self._scalar_mix(recombined_embeddings, input_mask)
        else:
            mix = recombined_embeddings[-1]

        # At this point, mix is (batch_size * d1 * ... * dn, sequence_length, embedding_dim)

        dims = initial_dims if needs_split else input_ids.size()
        return util.uncombine_initial_dims(mix, dims)

    @overrides
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super().state_dict(destination, prefix, keep_vars)

        keys_to_remove = []
        for key in self.untracked_keys:
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

