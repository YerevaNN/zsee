from collections import defaultdict, Counter

from torch import Tensor

from allennlp.modules import SimilarityFunction
from allennlp.training.metrics import Metric, Average

from typing import Any, Dict, List, Tuple, Union

ConfusionMatrix = Counter

LabellingsDict = Dict[Any, str]
LabellingsList = List[Tuple[Any, str]]
Labellings = Union[LabellingsDict, LabellingsList]


@Metric.register('precision_recall_fscore')
class PrecisionRecallFScore(Metric):
    def __init__(self,
                 labels: List[str] = None,
                 beta: float = 1,
                 prefix: str = '',
                 report_labelwise: bool = False) -> None:
        self._labels = labels
        self._beta = beta
        self._prefix = prefix
        self._report_labelwise = report_labelwise
        # Metrics are reported for each label, so we compute statistics
        # for each label separately.
        self._labelwise_confusion_matrices: Dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)

    def __call__(self,
                 batch_predictions: List[Labellings],
                 batch_gold_labels: List[Labellings],
                 **kwargs):
        for predictions, gold_labels in zip(batch_predictions,
                                            batch_gold_labels):
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
                        verbose: bool = True):
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
                   verbose: bool = True) -> Dict[str, float]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """

        # If we had no comparisons, report nothing.
        if not self._labelwise_confusion_matrices:
            # No need to handle `reset` flag as it is already in zero-state.
            return dict()

        averaged: ConfusionMatrix = ConfusionMatrix()
        for confusion_matrix in self._labelwise_confusion_matrices.values():
            averaged += confusion_matrix

        scores: Dict[str, Any] = dict()

        for label in self._labels:
            prefix = f'{self._prefix}labelwise/{label}'
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
