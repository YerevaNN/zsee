from typing import List, Union, Iterator

from torch.utils import data

from allennlp.data import Sampler, BatchSampler

_Sampler = Union[Sampler, data.Sampler]
_BatchSampler = Union[BatchSampler, data.BatchSampler]


class SubsetSampler(Sampler, data.Sampler):

    def __init__(self,
                 data_source: data.Subset,
                 sampler: _Sampler):
        self.data_source = data_source
        self.sampler = sampler

    def _map_indices_back(self, idx: int) -> int:
        return self.data_source.indices[idx]

    def __iter__(self) -> Iterator[Union[int, List[int]]]:
        for idx in iter(self.sampler):
            yield self._map_indices_back(idx)

    def __len__(self):
        return len(self.sampler)


class SubsetBatchSampler(SubsetSampler, BatchSampler):

    sampler: _BatchSampler

    def __init__(self,
                 data_source: data.Subset,
                 sampler: _BatchSampler):
        super().__init__(data_source,
                         sampler)

    @property
    def batch_size(self):
        return self.sampler.batch_size

    def _map_indices_back(self, idx: List[int]) -> List[int]:
        indices = []
        for sample_idx in idx:
            sample_idx = super()._map_indices_back(sample_idx)
            indices.append(sample_idx)
        return indices
