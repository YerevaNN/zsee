import logging
from collections import defaultdict
from typing import Dict, Any,  List

from overrides import overrides

import numpy as np
import torch
from torch import Tensor

from allennlp.common import Lazy
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from zsee.metrics import PrecisionRecallFScore, MultiClassConfusionMatrix, ReportSamplewiseTextClassification
from zsee.modules import ClassBalancedFocalLoss

from zsee.models.token_level import ZSEE

logger = logging.getLogger(__name__)


@Model.register('sentence_level_zsee', 'from_partial_objects')
class SentenceLevelZSEE(ZSEE):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 pooler: Seq2VecEncoder,
                 *,
                 logits_threshold: float = 0,
                 softmax: bool = False,
                 gamma: float = 0,
                 beta: float = 0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 report_confusion_matrix: bool = False,
                 report_samplewise: bool = False,
                 **kwargs) -> None:
        super().__init__(vocab,
                         text_field_embedder=text_field_embedder,
                         encoder=encoder,
                         **kwargs)

        self._pooler = pooler
        self._logits_threshold = logits_threshold
        self._softmax = softmax

        class_statistics = torch.ones(self._num_trigger_classes)
        vocab_counter: Dict[str, Dict[str, int]] = getattr(vocab, "_retained_counter", {})
        label_counter = vocab_counter.get(self._trigger_label_namespace)
        if label_counter is not None:
            for label, rank in label_counter.items():
                idx = vocab.get_token_index(label, namespace=self._trigger_label_namespace)
                class_statistics[idx] = rank
            logger.info(f'Class statistics: {class_statistics}')
        else:
            logger.info('The vocab counter is not retained.')

        if softmax:
            self._loss = ClassBalancedFocalLoss(CrossEntropyLoss,
                                                gamma=gamma,
                                                beta=beta,
                                                class_statistics=class_statistics)
        else:
            self._loss = ClassBalancedFocalLoss(BCEWithLogitsLoss,
                                                gamma=gamma,
                                                beta=beta,
                                                class_statistics=class_statistics[..., 1:])

        self._report_confusion_matrix = report_confusion_matrix
        self._report_samplewise = report_samplewise

        initializer(self)

        self.metrics: Dict[str, Dict[str, Metric]] = defaultdict(self._init_metrics)

    def _init_metrics(self) -> Dict[str, Metric]:
        prf = PrecisionRecallFScore(labels=self._labels,
                                    prefix='',
                                    verbose=False,
                                    report_labelwise=self._report_labelwise)
        confusion_matrix = MultiClassConfusionMatrix(labels=self._labels,
                                                     prefix='_figures/')
        samplewise = ReportSamplewiseTextClassification()

        return {
            'prf': prf,
            'confusion_matrix': confusion_matrix,
            'samplewise': samplewise
        }

    @classmethod
    def from_partial_objects(
            cls,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder: Lazy[Seq2SeqEncoder],
            pooler: Lazy[Seq2VecEncoder],
            logits_threshold: float = 0,
            softmax: bool = False,
            gamma: float = 0,
            beta: float = 0,
            initializer: InitializerApplicator = InitializerApplicator(),
            report_samplewise: bool = False,
            **kwargs
    ):
        mapping_dim = text_field_embedder.get_output_dim()
        encoder_ = encoder.construct(input_dim=mapping_dim)

        encoding_dim = encoder_.get_output_dim()
        pooler_ = pooler.construct(embedding_dim=encoding_dim)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder_,
                   pooler=pooler_,
                   logits_threshold=logits_threshold,
                   softmax=softmax,
                   gamma=gamma,
                   beta=beta,
                   initializer=initializer,
                   report_samplewise=report_samplewise,
                   **kwargs)

    @overrides
    def forward(self,
                text: Dict[str, Any],
                sentence_trigger_labels: Tensor = None,
                dataset: List[str] = None,
                **metadata) -> Dict[str, Any]:
        # Output dict to collect forward results
        output_dict: Dict[str, Any] = dict()

        # The raw tokens are stored for decoding
        # output_dict.update(metadata)

        # Shape: (batch_size, num_tokens, embedding_dim)
        contextual_embeddings = self._text_field_embedder(text)

        # Shape: (batch_size, num_tokens)
        mask = get_text_field_mask(text)
        # output_dict['mask'] = mask

        if mask.size(1) != contextual_embeddings.size(1):
            # mask = (text['bert']['input_ids'] != 0).long()
            mask = text['pretrained_transformer']['wordpiece_mask'].long()

        output_dict['mask'] = mask
        #
        # # Shape: (batch_size, embedding_dim)
        # sentence_embeddings = self._pooler(contextual_embeddings, mask=mask)

        # Apply normalization layer if needed
        if self._normalize:
            raise NotImplementedError

        output_dict['contextual_embeddings'] = contextual_embeddings
        # output_dict['sentence_embeddings'] = sentence_embeddings

        # Shape: (batch_size, embedding_dim)
        sentence_embeddings = self._embeddings_dropout(contextual_embeddings)

        if self._encoder is not None:
            # Shape: (batch_size, encoder_dim)
            hidden = self._encoder(sentence_embeddings)
        else:
            hidden = sentence_embeddings
        # output_dict['encoder_embeddings'] = hidden
        output_dict['encoded_embeddings'] = hidden
        hidden = self._dropout(hidden)

        # Shape: (batch_size, embedding_dim)
        hidden = self._pooler(hidden, mask=mask)
        output_dict['pooled_embeddings'] = hidden

        # Shape: (batch_size, num_trigger_classes)
        if self._projection is not None:
            logits = self._projection(hidden)
        else:
            assert hidden.size(-1) == self._num_trigger_classes
            logits = hidden
        output_dict['logits'] = logits

        if sentence_trigger_labels is None:
            return output_dict

        # # To make label-indexing consistent, we always use 0 for null label
        # # Pass non-null label logits only
        # probabilities = self._top_layer(logits)

        if self._softmax:
            loss = self._loss(logits,
                              sentence_trigger_labels.argmax(-1))
        else:
            loss = self._loss(logits[..., 1:],
                              sentence_trigger_labels[..., 1:].float())
        output_dict['loss'] = loss

        output_dict = self.decode(output_dict, False)

        predicted_labels_boolean = output_dict['predicted_labels_boolean']

        # Update Metrics
        if dataset is None:
            dataset = ['default']
        dataset, = set(dataset)

        # TODO Get rid of this dirty code
        self.metrics[dataset]['confusion_matrix'](predicted_labels_boolean.argmax(-1),
                                                  sentence_trigger_labels.argmax(-1).detach().cpu().numpy())

        self.metrics[dataset]['prf']([
            [
                ('sentence_wise', self._labels[label_idx])
                for label_idx, label_true
                in enumerate(sample_labels)
                if label_idx and label_true
            ]
            for sample_labels
            in predicted_labels_boolean.tolist()
        ], [
            [
                ('sentence_wise', self._labels[label_idx])
                for label_idx, label_true
                in enumerate(sample_labels)
                if label_idx and label_true
            ]
            for sample_labels
            in sentence_trigger_labels.tolist()
        ])

        self.metrics[dataset]['samplewise'](predicted_labels_boolean,
                                            sentence_trigger_labels,
                                            metadata['tokens'])

        return output_dict

    @overrides
    def decode(self,
               output_dict: Dict[str, Any],
               add_null_label: bool = False) -> Dict[str, Any]:
        # Shape: (batch_size, num_classes)
        batch_logits = output_dict['logits'].detach().cpu()

        batch_predicted_labels: List[List[str]] = []
        batch_predicted_labels_boolean: List[List[bool]] = []

        # Shape: (num_classes,)
        logits: Tensor
        for logits in batch_logits:
            predicted_labels: List[str] = []
            predicted_labels_boolean: List[bool] = [False] * len(self._labels)

            if self._softmax:
                idx: int = logits.argmax().item()
                label = self._labels[idx]
                predicted_labels_boolean[idx] = True
                if not idx and add_null_label:
                    predicted_labels.append(label)
            else:
                idx: int
                label: str
                for idx, label in enumerate(self._labels):
                    # Discard null label logits
                    if not idx:
                        continue
                    if logits[idx] > self._logits_threshold:
                        predicted_labels.append(label)
                        predicted_labels_boolean[idx] = True

                if not predicted_labels:
                    null_label = self._labels[0]
                    predicted_labels_boolean[0] = True
                    if add_null_label:
                        predicted_labels.append(null_label)

            batch_predicted_labels.append(predicted_labels)
            batch_predicted_labels_boolean.append(predicted_labels_boolean)

        output_dict['predicted_labels'] = batch_predicted_labels
        output_dict['predicted_labels_boolean'] = np.array(batch_predicted_labels_boolean)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if not reset and not self._verbose:
            return dict()

        scores: Dict[str, Any] = {}

        # TODO what about underscores?

        for group, metrics in self.metrics.items():
            group_scores: Dict[str, Any] = {}

            prf = metrics['prf'].get_metric(reset)
            group_scores.update(prf)

            if self._report_confusion_matrix:
                cm = metrics['cm'].get_metric(reset)
                group_scores.update(cm)

            if self._report_samplewise:
                samplewise = metrics['samplewise'].get_metric(reset)
                group_scores['samplewise'] = samplewise

            for key, value in group_scores.items():
                scores[f'./{group}/{key}'] = value

        return scores
