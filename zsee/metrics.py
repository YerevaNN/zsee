from typing import Any, Dict, List, Tuple, Union, Iterable

import torch

from collections import defaultdict, Counter

from overrides import overrides

from allennlp.data import Token
from allennlp.training.metrics import Metric

ConfusionMatrix = Counter

LabellingsDict = Dict[Any, str]
LabellingsList = List[Tuple[Any, str]]
Labellings = Union[LabellingsDict, LabellingsList]


@Metric.register('report_samplewise')
class ReportSamplewise(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def _report_line(self, line: str):
        self.lines.append(line)

    def sample_to_line(self,
                       sample_pred,
                       sample_true,
                       sample_metadata):
        raise NotImplementedError

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 metadata: List[Any],
                 **kwargs):
        predictions: List[Any] = predictions.tolist()
        gold_labels: List[Any] = gold_labels.tolist()

        for sample_pred, sample_true, sample_metadata in zip(predictions,
                                                             gold_labels,
                                                             metadata):
            line = self.sample_to_line(sample_pred, sample_true, sample_metadata)
            self._report_line(line)

    @overrides
    def get_metric(self, reset: bool):
        metric = {
            'type': 'blob',
            'dump': False,
            'lines': self.lines
        }

        if reset:
            self.reset()

        return metric

    @overrides
    def reset(self) -> None:
        self.lines = []


class ReportSamplewiseTextClassification(ReportSamplewise):

    def sample_to_line(self,
                       sample_pred: List[int],
                       sample_true: List[int],
                       sample_metadata: List[Token]):
        snt = ' '.join(str(tok) for tok in sample_metadata).replace('\t', '_TAB_')

        cells = []
        cells += ('+' if flag else '.' for flag in sample_pred)
        cells += ('+' if flag else '.' for flag in sample_true)
        cells.append(snt)

        return '\t'.join(cells)


@Metric.register('precision_recall_fscore')
class PrecisionRecallFScore(Metric):
    def __init__(self,
                 labels: List[str] = None,
                 beta: float = 1,
                 prefix: str = '',
                 report_labelwise: bool = False,
                 verbose: bool = True,
                 samplewise_statistics: bool = False) -> None:
        self._labels = labels
        self._beta = beta
        self._prefix = prefix
        self._report_labelwise = report_labelwise
        self._verbose = verbose
        # Metrics are reported for each label, so we compute statistics
        # for each label separately.
        self._labelwise_confusion_matrices: Dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)

        if samplewise_statistics:
            self._samplewise_statistics: Dict[str, Dict[str, ConfusionMatrix]] = dict()
        else:
            self._samplewise_statistics = None

    def __call__(self,
                 batch_predictions: List[Labellings],
                 batch_gold_labels: List[Labellings],
                 batch_metadata: List[Any] = None,
                 **kwargs):
        if batch_metadata is None:
            batch_metadata = [None for _ in batch_gold_labels]

        for predictions, gold_labels, metadata in zip(batch_predictions,
                                                      batch_gold_labels,
                                                      batch_metadata):
            # In case of dicts, convert them into compatible lists.
            if isinstance(predictions, dict):
                predictions = list(predictions.items())
            if isinstance(gold_labels, dict):
                gold_labels = list(gold_labels.items())
            # Now compare each sample separately.
            labelwise_confusion_matrices = self._compare_sample(predictions, gold_labels)
            # Accumulate into the state of the each label
            for label in self._labels:
                label_confusion_matrix = labelwise_confusion_matrices[label]
                self._labelwise_confusion_matrices[label] += label_confusion_matrix

            if self._samplewise_statistics is None:
                continue

            self._samplewise_statistics = labelwise_confusion_matrices

    def _compare_sample(self,
                        predictions: LabellingsList,
                        gold_labels: LabellingsList):
        # The implementation may seem little bit complicated,
        # but it was done to ensure also support of multi-label datapoints.

        # Now, collect predicted datapoints for each label.
        # `Counter`s are used as multisets.
        pred_datapoints: Dict[str, Counter[Any]] = defaultdict(Counter)
        gold_datapoints: Dict[str, Counter[Any]] = defaultdict(Counter)
        for datapoint, label in predictions:
            pred_datapoints[label][datapoint] += 1
        for datapoint, label in gold_labels:
            gold_datapoints[label][datapoint] += 1

        return {
            label: self._confusion_matrix(pred_datapoints[label],
                                          gold_datapoints[label])
            for label in self._labels
        }

    def _confusion_matrix(self,
                          predictions: Counter = None,
                          gold_labelled: Counter = None) -> ConfusionMatrix:
        if predictions is None:
            predictions = Counter()

        if gold_labelled is None:
            gold_labelled = Counter()

        # Now compare sets / multisets
        true_positive_datapoints = predictions & gold_labelled
        false_positive_datapoints = predictions - gold_labelled
        false_negative_datapoints = gold_labelled - predictions
        # Store only counts
        true_positives = sum(true_positive_datapoints.values())
        false_positives = sum(false_positive_datapoints.values())
        false_negatives = sum(false_negative_datapoints.values())

        return ConfusionMatrix(TP=true_positives, FP=false_positives,
                               FN=false_negatives)

    def _compute_scores(self,
                        confusion_matrix: ConfusionMatrix,
                        prefix: str = 'averaged',
                        verbose: bool = False):
        true_positives = confusion_matrix['TP']
        false_positives = confusion_matrix['FP']
        false_negatives = confusion_matrix['FN']

        gold_positives = true_positives + false_negatives
        predicted_positives = true_positives + false_positives

        if not predicted_positives:
            precision = 0
        else:
            precision = true_positives / predicted_positives

        if not gold_positives:
            recall = 0
        else:
            recall = true_positives / gold_positives

        if not precision and not recall:
            f_score = 0
        else:
            nom = (1 + self._beta ** 2) * precision * recall
            denom = (self._beta ** 2) * precision + recall
            f_score = nom / denom

        metrics = {
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall,

            f'{prefix}_f{self._beta}': f_score
        }

        if verbose:
            metrics.update({
                f'{prefix}_TP': true_positives,
                f'{prefix}_FP': false_positives,
                f'{prefix}_FN': false_negatives,
            })

        return metrics

    def get_metric(self,
                   reset: bool,
                   verbose: bool = None) -> Dict[str, float]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """

        if verbose is None:
            verbose = self._verbose

        # If we had no comparisons, report nothing.
        if not self._labelwise_confusion_matrices:
            # No need to handle `reset` flag as it is already in zero-state.
            return dict()

        averaged: ConfusionMatrix = ConfusionMatrix()
        for confusion_matrix in self._labelwise_confusion_matrices.values():
            averaged += confusion_matrix

        scores: Dict[str, Any] = dict()

        for label in self._labels:
            prefix = f'{self._prefix}_labelwise/{label}'
            label_scores = self._compute_scores(self._labelwise_confusion_matrices[label],
                                                prefix,
                                                verbose=verbose)
            if self._report_labelwise:
                scores.update(label_scores)

        prefix = f'{self._prefix}averaged'
        averaged_scores = self._compute_scores(averaged, prefix, verbose=verbose)
        scores.update(averaged_scores)

        if reset:
            self.reset()

        return scores

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self._labelwise_confusion_matrices.clear()


@Metric.register('multi_class_confusion_matrix')
class MultiClassConfusionMatrix(Metric):

    def __init__(self,
                 labels: List[str] = None,
                 beta: float = 1,
                 prefix: str = '') -> None:
        self._labels = labels
        self._beta = beta
        self._prefix = prefix
        self._confusion_matrix = torch.zeros(len(labels), len(labels), dtype=torch.int32)

    def __call__(self,
                 batch_predictions: Iterable[Union[int, str]],
                 batch_gold_labels: Iterable[Union[int, str]],
                 **kwargs):
        for prediction, gold_label in zip(batch_predictions, batch_gold_labels):
            if isinstance(prediction, str):
                prediction = self._labels.index(prediction)
            if isinstance(gold_label, str):
                gold_label = self._labels.index(gold_label)
            self._confusion_matrix[gold_label, prediction] += 1

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        # if not reset:
        #     return {}

        # fig = _figure(self._confusion_matrix, self._labels)

        metrics = {
            f'{self._prefix}confusion_matrix': {
                'type': 'confusion_matrix',
                'labels': self._labels,
                'confusion_matrix': self._confusion_matrix.tolist()
            }
        }

        if reset:
            self.reset()

        return metrics

    def reset(self) -> None:
        self._confusion_matrix.zero_()

#
# @Metric.register('multilabel_precision_recall_fscore')
# class MultilabelPrecisionRecallFScore(Metric):
#
#     def __init__(self,
#                  labels: List[str] = None,
#                  beta: float = 1,
#                  prefix: str = '',
#                  report_labelwise: bool = False,
#                  average: str = 'micro') -> None:
#         if report_labelwise:
#             raise NotImplementedError
#         self._labels = labels
#         self._beta = beta
#         self._prefix = prefix
#         self._average = average
#
#         self._predictions: List[Tensor] = []
#         self._targets: List[Tensor] = []
#
#     def __call__(self,
#                  predictions: Tensor,
#                  gold_labels: Tensor,
#                  **kwargs):
#         self._predictions.extend(predictions)
#         self._targets.extend(gold_labels)
#
#     def get_metric(self, reset: bool) -> Dict[str, float]:
#         # if not reset:
#         #     return dict()
#
#         metrics = precision_recall_fscore_support(y_true=self._targets,
#                                                   y_pred=self._predictions,
#                                                   beta=self._beta,
#                                                   labels=self._labels,
#                                                   average=self._average,
#                                                   zero_division=0)
#         precision, recall, fscore, _ = metrics
#
#         if reset:
#             self.reset()
#
#         return {
#             f'{self._prefix}averaged_precision': precision,
#             f'{self._prefix}averaged_recall': recall,
#             f'{self._prefix}averaged_f{self._beta}': fscore
#         }
#
#     def reset(self) -> None:
#         self._predictions = []
#         self._targets = []
