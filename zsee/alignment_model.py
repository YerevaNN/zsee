from typing import Dict, Any, Union, Tuple, List

from overrides import overrides

import torch
from torch import Tensor

from allennlp.data import Vocabulary, DataArray
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, SimilarityFunction, Seq2VecEncoder, TimeDistributed
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric, Average


@SimilarityFunction.register('mse')
class MSE(SimilarityFunction):

    def forward(self,
                tensor_1: torch.Tensor,
                tensor_2: torch.Tensor) -> torch.Tensor:
        difference = tensor_1 - tensor_2
        difference_square = difference * difference
        return difference_square.sum(-1)


@SimilarityFunction.register('minkowski_distance')
class MinkowskiDistance(SimilarityFunction):

    def __init__(self,
                 p: float = 2.,
                 eps: float = 1e-6,
                 keepdim: bool = False) -> None:
        super().__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        pass


class AlignmentMetric(Metric):
    def __init__(self,
                 distance: SimilarityFunction) -> None:
        self._distance = TimeDistributed(distance)
        self.num_correct = 0
        self.num_total = 0
        self.reset()

    def __call__(self,
                 source_embeddings: torch.Tensor,
                 target_embeddings: torch.Tensor):
        batch_size = source_embeddings.size(0)

        # Shape: (batch_size, 1, embedding_dim)
        source = source_embeddings.unsqueeze(1)

        # Shape: (1, batch_size, embedding_dim)
        target = target_embeddings.unsqueeze(0)

        # Shapes:
        # (batch_size, batch_size, embedding_dim),
        # (batch_size, batch_size, embedding_dim)
        source, target = torch.broadcast_tensors(source, target)

        # Shape: (batch_size, batch_size)
        distances = self._distance(source, target)
        # Shape: (batch_size,)
        closest = distances.argmin(-1).cpu()
        indices = torch.arange(batch_size)

        num_correct = torch.nonzero(closest == indices).size(0)

        self.num_correct += num_correct
        self.num_total += batch_size


    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        if self.num_total > 0:
            accuracy = self.num_correct / self.num_total
        else:
            accuracy = 0

        if reset:
            self.reset()

        return accuracy

    def reset(self):
        self.num_correct = 0
        self.num_total = 0


@Model.register('embeddings_alignment')
class AlignmentModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 distance: SimilarityFunction,
                 text_field_embedder: TextFieldEmbedder = None,
                 pooler: Seq2VecEncoder = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 map_both: bool = False,
                 models: List[Model] = None,
                 triplet_loss_margin: float = None,
                 verbose: bool = False) -> None:
        super().__init__(vocab)

        # This is hack to share
        if text_field_embedder is None:
            for model in models:
                text_field_embedder = getattr(model, '_text_field_embedder', text_field_embedder)
            assert text_field_embedder is not None
        if pooler is None:
            for model in models:
                pooler = getattr(model, '_pooler', pooler)
            assert pooler

        self._text_field_embedder = text_field_embedder
        self._pooler = pooler
        self._distance = distance
        self._alignment_accuracy = AlignmentMetric(distance)
        self._map_both = map_both
        self._triplet_loss_margin = triplet_loss_margin
        self._verbose = verbose

        self._num_samples_seen = 0
        self._average_distance = Average()
        self._average_pairwise_distance = Average()
        self._average_source_pairwise_distance = Average()
        self._average_target_pairwise_distance = Average()

        initializer(self)

    def forward_text(self,
                     text: Dict[str, DataArray],
                     mapped: bool = True) -> Tensor:
        # Shape: (batch_size, num_tokens)
        mask = get_text_field_mask(text)
        # Shape: (batch_size, num_tokens, embedding_dim)
        mapped_token_embeddings = self._text_field_embedder(text, mapped=mapped)
        # # Shape: (batch_size, embedding_dim)
        # source_embeddings: Tensor = self._pooler(source_token_embeddings,
        #                                          mask=source_mask)
        # Shape: (batch_size, embedding_dim)
        mapped_embeddings: Tensor = self._pooler(mapped_token_embeddings,
                                                 mask=mask)
        return mapped_embeddings

    @overrides
    def forward(self,
                source_snt: Dict[str, DataArray] = None,
                target_snt: Dict[str, DataArray] = None,
                **metadata) -> Dict[str, Any]:
        # If `source_snt` is not provided than report nothing.
        if source_snt is None:
            return None

        # Output dict to collect forward results
        output_dict: Dict[str, Any] = dict()
        # The raw tokens are stored for decoding
        output_dict.update(metadata)

        # # Shape: (batch_size, num_source_tokens)
        # source_mask = get_text_field_mask(source_snt)
        # # Shape: (batch_size, num_source_tokens, embedding_dim)
        # mapped_token_embeddings = self._text_field_embedder(source_snt)
        # # # Shape: (batch_size, embedding_dim)
        # # source_embeddings: Tensor = self._pooler(source_token_embeddings,
        # #                                          mask=source_mask)
        # # Shape: (batch_size, embedding_dim)
        # mapped_embeddings: Tensor = self._pooler(mapped_token_embeddings,
        #                                          mask=source_mask)

        # Shape: (batch_size, embedding_dim)
        source_embeddings = self.forward_text(source_snt)

        batch_size = source_embeddings.size(0)

        if target_snt is None:
            return output_dict

        # # Shape: (batch_size, num_source_tokens)
        # target_mask = get_text_field_mask(target_snt)
        # # Shape: (batch_size, num_target_tokens, embedding_dim)
        # target_token_embeddings = self._text_field_embedder(target_snt, mapped=False)
        # # Shape: (batch_size, embedding_dim)
        # target_embeddings: Tensor = self._pooler(target_token_embeddings,
        #                                          mask=target_mask)

        # Shape: (batch_size, embedding_dim)
        target_embeddings = self.forward_text(target_snt, mapped=self._map_both)

        # Shape: (batch_size,)
        distance = self._distance(source_embeddings, target_embeddings)
        # Shape: (batch_size, batch_size)
        pariwise_distances = self._distance(source_embeddings.unsqueeze(0),
                                            target_embeddings.unsqueeze(1))
        # Shape: (batch_size)
        random_pair_distances = pariwise_distances[torch.arange(batch_size),
                                                   torch.randperm(batch_size)]
        # Shape: (batch_size, batch_size)
        source_pairwise_distances = self._distance(source_embeddings.unsqueeze(0),
                                                   source_embeddings.unsqueeze(1))
        # Shape: (batch_size, batch_size)
        target_pairwise_distances = self._distance(source_embeddings.unsqueeze(0),
                                                   source_embeddings.unsqueeze(1))

        positive_loss: Tensor = distance.mean()
        output_dict['positive_loss'] = positive_loss
        if self._triplet_loss_margin is None:
            # Shape: ()
            output_dict['loss'] = positive_loss
        else:
            negative_loss: Tensor = random_pair_distances.mean() - self._triplet_loss_margin
            output_dict['negative_loss'] = negative_loss
            output_dict['loss'] = (positive_loss - negative_loss).clamp(min=0)

        with torch.no_grad():
            self._num_samples_seen += 1

            self._alignment_accuracy(source_embeddings, target_embeddings)

            self._average_distance(distance.mean().item())
            self._average_pairwise_distance(pariwise_distances.mean().item())
            self._average_source_pairwise_distance(source_pairwise_distances.mean().item())
            self._average_target_pairwise_distance(target_pairwise_distances.mean().item())

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        verbose = self._verbose or reset

        if not verbose:
            return dict()

        if not self._num_samples_seen:
            return dict()

        return {
            'alignment_accuracy': self._alignment_accuracy.get_metric(reset),
            'average_distance': self._average_distance.get_metric(reset),
            'average_pairwise_distance': self._average_pairwise_distance.get_metric(reset),
            'average_source_pairwise_distance': self._average_source_pairwise_distance.get_metric(reset),
            'average_target_pairwise_distance': self._average_target_pairwise_distance.get_metric(reset)
        }
