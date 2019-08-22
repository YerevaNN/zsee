from typing import Dict, Any

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


@Model.register('zsee')
class ZSEE(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 event_label_namespace: str = 'event_labels') -> None:
        super().__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._event_label_namespace = event_label_namespace

        num_event_classes = vocab.get_vocab_size(event_label_namespace)
        self._projection = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                           out_features=num_event_classes)

        self._accuracy = CategoricalAccuracy()

    def forward(self,
                text: Dict[str, Any],
                event_labels: torch.LongTensor = None,
                **metadata) -> Dict[str, Any]:
        # Output dict to collect forward results
        output_dict: Dict[str, Any] = dict()

        # Shape: (batch_size, num_tokens, ??)
        mask = get_text_field_mask(text)
        output_dict['mask'] = mask

        # Shape: (batch_size, num_tokens, embedding_dim)
        text_embeddings = self._text_field_embedder(text)

        # Shape: (batch_size, num_tokens, encoder_dim)
        hidden = self._encoder(text_embeddings, mask)

        tag_logits = self._projection(hidden)
        output_dict['tag_logits'] = tag_logits

        if event_labels is None:
            return output_dict

        loss = sequence_cross_entropy_with_logits(tag_logits, event_labels, mask)
        output_dict["loss"] = loss

        # Computing metrics

        self._accuracy(tag_logits, event_labels, mask)
        # Decode
        output_dict = self.decode(output_dict)



        return output_dict




    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        if 'predicted_tags' in output_dict:
            return output_dict



        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset)
        }
