from typing import Dict, Any, Tuple, List, Union, Iterable

import torch
from allennlp.data import Vocabulary, Token
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import Dropout, Linear

from .metrics import PrecisionRecallFScore


@Model.register('zsee')
class ZSEE(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 embeddings_dropout: float = 0,
                 dropout: float = 0,
                 verbose: Union[bool, Iterable[str]] = False,
                 balance: bool = False,
                 trigger_label_namespace: str = 'event_labels') -> None:
        super().__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._embeddings_dropout = Dropout(embeddings_dropout)
        self._dropout = Dropout(dropout)
        self._verbose = verbose
        self._balance = balance
        self._trigger_label_namespace = trigger_label_namespace

        num_trigger_classes = vocab.get_vocab_size(trigger_label_namespace)
        self._projection = Linear(in_features=encoder.get_output_dim(),
                                  out_features=num_trigger_classes)

        self._accuracy = CategoricalAccuracy()
        labels = vocab.get_token_to_index_vocabulary(self._trigger_label_namespace)

        # We have two (slight different) metric sets: char-based and token-based
        # Char-based metrics also capture error propagated by NER.
        # Token-based metrics are computed as well to:
        #  1. Measure difference as error we have because of NER,
        #  2. Compare with most of the previous work evaluated token-level,
        #  3. As fallback if our tokenization option does not provide token-char mappings.
        self._prf_char_seqs = PrecisionRecallFScore(labels=list(labels))
        self._prf_token_seqs = PrecisionRecallFScore(labels=list(labels),
                                                     prefix='token_level/')

    def _balancing_weights(self,
                           labels: torch.Tensor,
                           mask: torch.Tensor = None):
        # Build weight-mask for importance weighting
        # in case of imbalanced number of the labels.

        # `labels`
        # Shape: (batch_size, T_1, ..., T_n)
        # `mask`
        # Shape: (batch_size, T_1, ..., T_n)

        # Compute statistics for each label in the batch
        # Shape: (num_labels,)
        if mask is None:
            mask = torch.ones_like(labels)

        mask = mask.float()

        # Only 1-D tensors are supported by `bincounts`, we need to flatten them
        num_label_occurrences = labels.flatten().bincount(mask.flatten())
        target_weight = mask.sum()

        # Clamp the tensor to avoid division-by-zero
        num_label_occurrences = num_label_occurrences.clamp(min=1)

        weights = mask / num_label_occurrences[labels]
        resulted_weight = weights.sum()

        return weights / resulted_weight * target_weight

    def forward(self,
                text: Dict[str, Any],
                trigger_labels: torch.LongTensor = None,
                **metadata) -> Dict[str, Any]:
        # Output dict to collect forward results
        output_dict: Dict[str, Any] = dict()

        # The raw tokens are stored for decoding
        output_dict.update(metadata)

        # Shape: (batch_size, num_tokens)
        mask = get_text_field_mask(text)
        output_dict['mask'] = mask

        # Shape: (batch_size, num_tokens, embedding_dim)
        text_embeddings = self._text_field_embedder(text)
        text_embeddings = self._embeddings_dropout(text_embeddings)

        # Shape: (batch_size, num_tokens, encoder_dim)
        hidden = self._encoder(text_embeddings, mask)
        hidden = self._dropout(hidden)

        # Shape: (batch_size, num_tokens, num_trigger_classes)
        tag_logits = self._projection(hidden)
        output_dict['tag_logits'] = tag_logits

        if trigger_labels is None:
            return output_dict

        balancing_weights = self._balancing_weights(trigger_labels,
                                                    mask=mask)
        loss = sequence_cross_entropy_with_logits(tag_logits, trigger_labels,
                                                  balancing_weights)
        output_dict['loss'] = loss

        # Computing metrics

        self._accuracy(tag_logits, trigger_labels, mask)
        # Decode
        # if self.training:
        #     return output_dict

        output_dict = self.decode(output_dict)

        # Token-level computation of predictions
        self._prf_token_seqs(output_dict['pred_trigger_token_seqs'],
                             metadata['trigger_token_seqs'])

        # If mapping was provided and model was able to decode char-based
        # trigger span boundaries, then compute metrics also w.r.t. them
        if 'trigger_char_seqs' in metadata:
            self._prf_char_seqs(output_dict['pred_trigger_char_seqs'],
                                metadata['trigger_char_seqs'])

        return output_dict

    class Char2TokenMappingMissing(Exception):
        pass

    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        batch_tokens: List[List[Token]] = output_dict['tokens']
        # Shape: (batch_size, num_tokens, num_trigger_classes)
        tag_logits: torch.Tensor = output_dict['tag_logits'].detach().cpu().numpy()
        # Shape: (batch_size, num_tokens)
        batch_pred_tags = tag_logits.argmax(-1)
        # Shape: (batch_size, num_tokens)
        batch_mask: torch.Tensor = output_dict['mask']

        # First, decode token-based trigger spans.
        batch_pred_trigger_token_seqs = self._decode_trigger_token_seqs(batch_pred_tags,
                                                                        batch_mask)
        output_dict['pred_trigger_token_seqs'] = batch_pred_trigger_token_seqs

        # Try to decode char-based boundaries of the trigger span.
        # If no mapping is provided, skip the step.
        try:
            batch_pred_trigger_char_seqs = self._decode_trigger_char_seqs(batch_tokens,
                                                                          batch_pred_trigger_token_seqs)
            output_dict['pred_trigger_char_seqs'] = batch_pred_trigger_char_seqs
        except ZSEE.Char2TokenMappingMissing:
            output_dict['pred_trigger_char_seqs'] = None

        return output_dict

    def _decode_trigger_token_seqs(self,
                                   batch_pred_tags: torch.Tensor,
                                   batch_mask: torch.Tensor
                                   ) -> List[List[Tuple[Tuple[int, int], str]]]:

        batch_predicted_triggers: List[List[Tuple[Tuple[int, int], str]]] = []
        for predicted_tags, mask in zip(batch_pred_tags,
                                        batch_mask):
            # First, we decode token-level spans of the mentioned events
            predicted_triggers: Dict[Tuple[int, int], str] = dict()
            # Token-based offset of the start, inclusive
            first_idx = 0
            # Token-based offset of the end, inclusive
            last_idx = 0
            # If two consecutive labels are the same, then we need to join them
            previous_label = 'O'

            for idx, (tag_idx, to_process) in enumerate(zip(predicted_tags, mask)):
                if not to_process:
                    continue
                label = self.vocab.get_token_from_index(tag_idx,
                                                        self._trigger_label_namespace)
                # If new label is occurred, then set the new start
                if label != previous_label:
                    first_idx = idx

                # Discard the mention
                if (first_idx, last_idx) in predicted_triggers:
                    del predicted_triggers[first_idx, last_idx]

                # Define extended mention span
                last_idx = idx
                previous_label = label

                # Register if has label
                if label is 'O':
                    continue
                predicted_triggers[first_idx, last_idx] = label

            # Convert dicts into lists
            batch_predicted_triggers.append([
                (span, label)
                for span, label
                in predicted_triggers.items()
            ])

        return batch_predicted_triggers

    def _decode_trigger_char_seqs(self,
                                  batch_tokens: List[List[Token]],
                                  batch_pred_trigger_token_seqs: List[List[Tuple[Tuple[int, int], str]]]
                                  ) -> List[List[Tuple[Tuple[int, int], str]]]:

        batch_predicted_triggers: List[List[Tuple[Tuple[int, int], str]]] = []
        for tokens, trigger_token_spans in zip(batch_tokens, batch_pred_trigger_token_seqs):
            # Now, we are ready to map token spans back to raw text to get
            # char-based mention boundaries
            predicted_triggers: List[Tuple[Tuple[int, int], str]] = []
            for (first_idx, last_idx), label in trigger_token_spans:
                first = tokens[first_idx]
                last = tokens[last_idx]

                if getattr(first, 'idx', None) is None:
                    raise ZSEE.Char2TokenMappingMissing('No char2token mapping is provided.')

                # Char-based start offset, inclusive
                start = first.idx
                # Char-based end offset, exclusive
                end = last.idx + len(last.text)
                # Report the predicted trigger
                trigger = (start, end), label
                predicted_triggers.append(trigger)

            batch_predicted_triggers.append(predicted_triggers)

        return batch_predicted_triggers

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if not reset and not self._verbose:
            return dict()

        # TODO Show only keys of `self._verbose`

        prf_char_seqs = self._prf_char_seqs.get_metric(reset)
        prf_token_seqs = self._prf_token_seqs.get_metric(reset)

        scores = {
            'accuracy': self._accuracy.get_metric(reset),
            **prf_char_seqs,
            **prf_token_seqs
        }

        return scores
