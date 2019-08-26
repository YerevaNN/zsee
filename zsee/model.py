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
                 event_label_namespace: str = 'event_labels') -> None:
        super().__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._embeddings_dropout = Dropout(embeddings_dropout)
        self._dropout = Dropout(dropout)
        self._verbose = verbose
        self._balance = balance
        self._event_label_namespace = event_label_namespace

        num_event_classes = vocab.get_vocab_size(event_label_namespace)
        self._projection = Linear(in_features=encoder.get_output_dim(),
                                  out_features=num_event_classes)

        self._accuracy = CategoricalAccuracy()
        labels = vocab.get_token_to_index_vocabulary(self._event_label_namespace)
        self._precision_recall_fscore = PrecisionRecallFScore(labels=list(labels))
        # self._precision_recall_fscore_tokens = PrecisionRecallFScore(labels=list(labels))

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
                event_labels: torch.LongTensor = None,
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

        # Shape: (batch_size, num_tokens, num_event_types)
        tag_logits = self._projection(hidden)
        output_dict['tag_logits'] = tag_logits

        if event_labels is None:
            return output_dict

        balancing_weights = self._balancing_weights(event_labels,
                                                    mask=mask)
        loss = sequence_cross_entropy_with_logits(tag_logits, event_labels,
                                                  balancing_weights)
        output_dict['loss'] = loss

        # Computing metrics

        self._accuracy(tag_logits, event_labels, mask)
        # Decode
        # if self.training:
        #     return output_dict

        output_dict = self.decode(output_dict)
        # Compute Metrics on char-based decoded spans
        batch_predicted_events = output_dict['predicted_events']
        batch_raw_sentence_events = metadata['raw_sentence_events']
        self._precision_recall_fscore(batch_predicted_events,
                                      batch_raw_sentence_events)
        # # Token-level computation of predictions
        # batch_predicted_events_tokens = output_dict['predicted_events_tokens']
        # batch_sentence_events = metadata['sentence_events']
        # self._precision_recall_fscore_tokens(batch_predicted_events_tokens,
        #                                      batch_sentence_events)

        return output_dict

    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        if 'predicted_events' in output_dict:
            return output_dict

        batch_tokens: List[List[Token]] = output_dict['tokens']
        # Shape: (batch_size, num_tokens, num_event_types)
        tag_logits: torch.Tensor = output_dict['tag_logits'].detach().cpu().numpy()
        # Shape: (batch_size, num_tokens)
        batch_predicted_tags = tag_logits.argmax(-1)
        # Shape: (batch_size, num_tokens)
        batch_mask: torch.Tensor = output_dict['mask']

        batch_predicted_events_tokens: List[Dict[Tuple[int, int], str]] = []
        batch_predicted_events: List[Dict[Tuple[int, int], str]] = []
        for predicted_tags, mask, tokens in zip(batch_predicted_tags,
                                                batch_mask,
                                                batch_tokens):
            # First, we decode token-level spans of the mentioned events
            event_token_spans: Dict[Tuple[int, int], str] = dict()
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
                                                        self._event_label_namespace)
                # If new label is occurred, then set the new start
                if label != previous_label:
                    first_idx = idx

                # Discard the mention
                if (first_idx, last_idx) in event_token_spans:
                    del event_token_spans[first_idx, last_idx]

                # Define extended mention span
                last_idx = idx
                previous_label = label

                # Register if has label
                if label is 'O':
                    continue
                event_token_spans[first_idx, last_idx] = label

            # Now, we are ready to map token spans back to raw text to get
            # char-based mention boundaries
            predicted_events: Dict[Tuple[int, int], str] = dict()
            for (first_idx, last_idx), label in event_token_spans.items():
                first = tokens[first_idx]
                last = tokens[last_idx]

                # Char-based start offset, inclusive
                start = first.idx
                # Char-based end offset, exclusive
                end = last.idx + len(last.text)
                # Report the predicted mention
                predicted_events[start, end] = label

            batch_predicted_events_tokens.append(event_token_spans)
            batch_predicted_events.append(predicted_events)

        output_dict['predicted_events_tokens'] = batch_predicted_events_tokens
        output_dict['predicted_events'] = batch_predicted_events

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if not reset and not self._verbose:
            return dict()

        # TODO Show only keys of `self._verbose`

        precision_recall_fscore = self._precision_recall_fscore.get_metric(reset)
        scores = {
            'accuracy': self._accuracy.get_metric(reset),
            **precision_recall_fscore
        }

        # for key, value in self._precision_recall_fscore_tokens.get_metric(reset).items():
        #     scores[f'token-level/{key}'] = value

        return scores
