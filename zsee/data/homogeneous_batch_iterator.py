from typing import Iterable, Dict, List
import random
from collections import defaultdict

from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator


@DataIterator.register("homogeneous_batch", exist_ok=True)
class HomogeneousBatchIterator(DataIterator):

    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 partition_key: str = "dataset",
                 skip_smaller_batches: bool = False,
                 until_finished: str = "all",
                 hops: Dict[str, int] = None) -> None:
        super().__init__(batch_size, instances_per_epoch, max_instances_in_memory,
                         cache_instances, track_epoch)
        self._partition_key = partition_key
        self._skip_smaller_batches = skip_smaller_batches
        self._until_finished = until_finished
        if hops is None:
            hops: Dict[str, int] = dict()
        self._hops = hops

    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        return super().get_num_batches(instances)

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)

            # Divvy up the instances based on their value of the "partition_key" field.
            hoppers: Dict[str, List[Instance]] = defaultdict(list)
            for instance in instance_list:
                partition_field: MetadataField = instance.fields.get(self._partition_key)  # type: ignore
                partition = getattr(partition_field, 'metadata', 'default')
                hoppers[partition].append(instance)

            # Get a `lazy_groups_of` iterator over each set of homogeneous instances.
            batches = {key: lazy_groups_of(iter(hopper), self._batch_size) for key, hopper in hoppers.items()}

            remaining = set(batches)
            finish_len = len(remaining) if self._until_finished == 'any' else 1

            # Yield batches in a round-robin fashion until none are left.
            while len(remaining) >= finish_len:
                for key, lazy_batches in batches.items():
                    hops = self._hops.get(key, 1)
                    for hop in range(hops):
                        try:
                            batch = next(lazy_batches)
                            if not self._skip_smaller_batches or len(batch) == self._batch_size:
                                yield Batch(batch)
                        except StopIteration:
                            remaining.discard(key)
