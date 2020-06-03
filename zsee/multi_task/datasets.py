from collections import defaultdict
from typing import Dict, List

from torch.utils.data import Subset, Dataset

from allennlp.data import Instance


class HomogeneousDataset(Subset):

    @classmethod
    def partition_data_source(cls,
                              data_source: Dataset,
                              partition_key: str
                              ) -> Dict[str, 'HomogeneousDataset']:

        partition_indices: Dict[str, List[int]] = defaultdict(list)

        num_total_samples = len(data_source)
        for idx in range(num_total_samples):
            instance: Instance = data_source[idx]
            partition = instance.fields[partition_key].metadata
            partition_indices[partition].append(idx)

        # TODO add checks
        return {
            partition: cls(data_source, indices)
            for partition, indices
            in partition_indices.items()
        }
