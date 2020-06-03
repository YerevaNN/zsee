import itertools
import logging
from typing import Dict, Iterable, List, Union, Set

from torch.utils import data

from allennlp.common.checks import ConfigurationError
from allennlp.data.samplers import BatchSampler

from allennlp.common.lazy import Lazy

from .datasets import HomogeneousDataset
from .samplers import SubsetBatchSampler

from .task_switchers import TaskSwitcher

logger = logging.getLogger(__name__)


_Sampler = Union[BatchSampler, data.Sampler]
_BatchSampler = Union[BatchSampler, data.BatchSampler]


@BatchSampler.register("homogeneous", "from_partial_objects")
class HomogeneousBatchSampler(BatchSampler):

    def __init__(
        self,
        data_sources: Dict[str, data.Dataset],
        task_switcher: TaskSwitcher,
        batch_samplers: Dict[str, _BatchSampler],
        num_loops_per_epoch: float = 1
    ):
        self.data_sources = data_sources
        self.task_switcher = task_switcher
        self.batch_samplers = batch_samplers
        self.num_loops_per_epoch = num_loops_per_epoch

    def _infinite_iter(self, task: str):
        iterable = self.batch_samplers[task]

        is_first_time = True
        while True:
            if not is_first_time:
                logger.debug(f'Reiterating over data of the task `{task}`')
            yield from iterable
            is_first_time = False

    def _iter(self) -> Iterable[List[int]]:
        iterators = {
            task: self._infinite_iter(task)
            for task in self.data_sources
        }

        for task in self.task_switcher:
            instance = next(iterators[task])
            yield instance

    def __iter__(self) -> Iterable[List[int]]:
        epoch_iterator = self._iter()
        num_loop_iters = len(self)
        yield from itertools.islice(epoch_iterator, num_loop_iters)

    def __len__(self):
        num_epoch_iters = len(self.task_switcher)
        num_loop_iters = int(num_epoch_iters // self.num_loops_per_epoch)
        return num_loop_iters

    @classmethod
    def from_partial_objects(
        cls,
        data_source: data.Dataset,
        task_switcher: Lazy[TaskSwitcher],
        task_batch_samplers: Dict[str, Lazy[BatchSampler]] = None,
        batch_sampler: Lazy[BatchSampler] = None,
        partition_key: str = 'dataset',
        default: str = None,
        batch_size: int = None,
        num_epoch_iterations: int = None,
        num_loops_per_epoch: int = 1
    ) -> "HomogeneousBatchSampler":

        data_sources = HomogeneousDataset.partition_data_source(data_source,
                                                                partition_key)

        if batch_sampler is not None:
            if task_batch_samplers is not None:
                raise ConfigurationError("Only one of `batch_sampler` and `task_batch_samplers` "
                                         "can be provided.")
            else:
                if default is None:
                    default = 'default'
                task_batch_samplers: Dict[str, Lazy[BatchSampler]] = {
                    'default': batch_sampler
                }
        else:
            if task_batch_samplers is None:
                raise ConfigurationError("No `batch_sampler` or `task_batch_samplers` "
                                         "was provided.")

        if default is None and len(task_batch_samplers) == 1:
            default, = task_batch_samplers

        used_samplers: Set[str] = set()
        lazy_batch_samplers: Dict[str, Lazy[BatchSampler]] = {}
        for partition in data_sources:
            if partition in task_batch_samplers:
                lazy_batch_sampler = task_batch_samplers[partition]
                used_samplers.add(partition)
            elif default in task_batch_samplers:
                lazy_batch_sampler = task_batch_samplers[default]
                used_samplers.add(default)
                logger.info(f"Using default sampler for the partition `{partition}`")
            else:
                raise ConfigurationError(f"No sampler (or default one) was provided "
                                         f"for the partition `{partition}`")
            lazy_batch_samplers[partition] = lazy_batch_sampler

        unused_samplers = task_batch_samplers.keys() - used_samplers
        if unused_samplers:
            logger.warning(f"Samplers `{unused_samplers}` were not used")

        #
        batch_sampler_extras = {}
        if batch_size is not None:
            batch_sampler_extras["batch_size"] = batch_size

        task_batch_samplers_: Dict[str, BatchSampler] = {}
        for partition in data_sources:
            lazy_sampler = lazy_batch_samplers[partition]
            data_source = data_sources[partition]
            # Batch samplers better never see any data_source
            sampler: BatchSampler = lazy_sampler.construct(data_source=data_source,
                                                           **batch_sampler_extras)
            task_batch_samplers_[partition] = SubsetBatchSampler(data_source=data_source,
                                                                 sampler=sampler)

        task_switcher = task_switcher.construct(task_samplers=task_batch_samplers_,
                                                num_epoch_iterations=num_epoch_iterations)  # TODO

        return cls(data_sources=data_sources,
                   task_switcher=task_switcher,
                   batch_samplers=task_batch_samplers_,
                   num_loops_per_epoch=num_loops_per_epoch)
