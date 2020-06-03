from typing import Iterable, Iterator, Tuple, Dict, List, Union

import itertools

import logging
import numpy as np
from overrides import overrides

from torch.utils import data

from allennlp.common import Registrable

logger = logging.getLogger(__name__)


class TaskSwitcher(Registrable):

    default_implementation = "chain"  # TODO implement chain

    def __init__(self,
                 task_samplers: Dict[str, data.BatchSampler],
                 num_epoch_iterations: int = None):
        self.task_samplers = task_samplers
        self._num_epoch_iterations = num_epoch_iterations
        # Prepare data sizes to calculate sampling rates
        self._est_num_task_samples = None

        self.num_tasks = len(task_samplers)
        self.tasks: List[str] = list(task_samplers.keys())
        self.samplers: List[data.BatchSampler] = list(task_samplers.values())

        # Shape: (num_tasks,)
        self.num_task_batches: List[int] = [
            len(batch_sampler) for batch_sampler in self.samplers
        ]

        logger.info(f'Tasks: {self.tasks}')
        logger.info(f'num_batches: {self.num_task_batches}')

    @property
    def num_epoch_iterations(self):
        if self._num_epoch_iterations is None:
            self._num_epoch_iterations = self._infer_epoch_iterations()
        return self._num_epoch_iterations

    def _infer_epoch_iterations(self) -> int:
        raise NotImplementedError

    def _yield_indices(self) -> Iterator[int]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        indices = self._yield_indices()
        for idx in itertools.islice(indices, self.num_epoch_iterations):
            yield self.tasks[idx]

    def __len__(self) -> int:
        return self.num_epoch_iterations

    @property
    def est_num_task_samples(self) -> List[int]:
        if self._est_num_task_samples is None:

            self._est_num_task_samples = [
                len(sampler) * getattr(sampler, "batch_size")
                for sampler in self.samplers
            ]

        # Shape: (num_tasks,)
        return self._est_num_task_samples


@TaskSwitcher.register('chain')
class ChainTaskSwitcher(TaskSwitcher):

    def __init__(self,
                 task_samplers: Dict[str, data.BatchSampler],
                 num_epoch_iterations: int = None):
        if num_epoch_iterations is not None:
            raise NotImplementedError('ChainTaskSwitcher does not support '
                                      'custom number of iterations.')
        super().__init__(task_samplers, num_epoch_iterations)

    def _infer_epoch_iterations(self) -> int:
        return sum(self.num_task_batches)

    def _yield_indices(self) -> Iterator[int]:
        for idx, num_batches in enumerate(self.num_task_batches):
            yield from itertools.repeat(idx, times=num_batches)


@TaskSwitcher.register('multihop')
class MultihopTaskSwitcher(TaskSwitcher):

    def __init__(self,
                 hops: Union[Dict[str, int], int] = None,
                 **kwargs):
        super().__init__(**kwargs)

        default_hops = 1
        if isinstance(hops, int):
            default_hops = hops

        if not isinstance(hops, dict):
            hops: Dict[str, int] = {}

        hops: Dict[str, int] = {
            task: hops.get(task, default_hops)
            for task in self.tasks
        }
        self.hops = hops

    def _infer_epoch_iterations(self) -> int:
        return sum([num_batches
                   for task, num_batches
                   in zip(self.tasks, self.num_task_batches)
                   if self.hops[task] > 0])

    def _yield_indices(self) -> Iterator[int]:
        num_batches_left = self.num_task_batches.copy()
        for task_id, task in itertools.cycle(enumerate(self.tasks)):
            hops = self.hops[task]
            for hop in range(hops):
                if not num_batches_left[task_id]:
                    break
                yield task_id
                num_batches_left[task_id] -= 1


class TaskSampler(TaskSwitcher):

    def __init__(self,
                 replacement: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.replacement = replacement
        # Shape: (num_tasks,)
        self.sampling_rates = self._normalized_sampling_rates()

    def _sampling_rates(self) -> Iterable[float]:
        raise NotImplementedError

    def _infer_epoch_iterations(self) -> int:
        # Shape: (num_tasks,)
        tasks_eta: np.ndarray = self.num_task_batches / self.sampling_rates
        # TODO maybe ceil to include the last batch?
        return tasks_eta.astype(np.int).min()

    def _scaled_sampling_rates(self) -> np.ndarray:
        # Shape: (num_tasks,)
        sampling_weights = self._sampling_rates()
        if not isinstance(sampling_weights, np.ndarray):
            sampling_weights = np.array(sampling_weights, dtype=np.float)
        return sampling_weights

    def _normalized_sampling_rates(self) -> np.ndarray:
        # Shape: (num_tasks,)
        sampling_weights = self._scaled_sampling_rates()
        # Shape: (num_tasks,)
        return sampling_weights / sampling_weights.sum()

    def _ps_with_replacement(self) -> Tuple[np.ndarray, np.ndarray]:
        # Shape: (num_tasks,)
        events = np.arange(self.num_tasks)
        # Shape: (num_tasks,)
        probabilities = self.sampling_rates

        return events, probabilities

    def _ps_without_replacement(self) -> Tuple[np.ndarray, np.ndarray]:
        # Shape: (num_tasks,)
        num_batches_sampled = self.num_epoch_iterations * self.sampling_rates
        num_batches_sampled = num_batches_sampled.astype(np.int)
        num_total_batches = num_batches_sampled.sum()

        # Now we have spread almost all batches out of `num_epoch_iterations`
        # However, there still may be some leftovers
        assert num_total_batches <= self.num_epoch_iterations
        # TODO find a correct way
        num_leftovers = self.num_epoch_iterations - num_total_batches
        candidates = np.flatnonzero(num_batches_sampled < self.num_task_batches)
        leftovers = np.random.choice(candidates, size=num_leftovers, replace=False)
        num_batches_sampled[leftovers] += 1
        num_total_batches = num_batches_sampled.sum()
        assert num_total_batches >= self.num_epoch_iterations

        # Shape: (num_tasks,)
        indices = np.arange(self.num_tasks)

        # Shape: (num_tasks,)
        batch_sampling_rates = np.ones()

        # Shape: (max_total_batches,)
        events = np.repeat(indices, num_batches_sampled)
        # Shape: (max_total_batches,)
        probabilities = np.repeat(batch_sampling_rates, num_batches_sampled)

        return events, probabilities

    def probability_space(self):
        if self.replacement:
            # The code for `without replacement` will work in this case
            # too, but this specific implementation is much faster.
            return self._ps_with_replacement()
        else:
            return self._ps_without_replacement()

    def _yield_indices(self) -> Iterator[int]:
        events, probabilities = self.probability_space()
        # Shape: (num_epoch_iterations,)
        task_indices = np.random.choice(events,
                                        p=probabilities,
                                        size=self.num_epoch_iterations,
                                        replace=self.replacement)
        yield from task_indices


@TaskSwitcher.register('uniform_sampler')
class UniformTaskSampler(TaskSampler):

    @overrides
    def _sampling_rates(self) -> Iterable[float]:
        return np.ones(self.num_tasks, dtype=np.float)


@TaskSwitcher.register('proportional_sampler')
@TaskSwitcher.register('linear_sampler')
class ProportionalTaskSampler(TaskSampler):

    def __init__(self,
                 inverse: bool = False,
                 log: bool = False,
                 compare_num_samples: bool = False,
                 clamp_min: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.inverse = inverse
        self.log = log
        self.compare_num_samples = compare_num_samples
        self.clamp_min = clamp_min

    @overrides
    def _sampling_rates(self) -> np.ndarray:
        # Shape: (num_tasks,)
        if self.compare_num_samples:
            sample_rates = self.est_num_task_samples
        else:
            sample_rates = self.num_task_batches

        return np.array(sample_rates, dtype=np.float).clip(min=self.clamp_min)

    @overrides
    def _scaled_sampling_rates(self) -> np.ndarray:
        sampling_rates = super()._scaled_sampling_rates()
        if self.log:
            sampling_rates = np.log(sampling_rates)
        if self.inverse:
            sampling_rates = 1 / sampling_rates
        return sampling_rates


@TaskSwitcher.register('pow_proportional_sampler')
@TaskSwitcher.register('pow_sampler')
class PowProportionalTaskSampler(ProportionalTaskSampler):

    def __init__(self,
                 power: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.power = power

    @overrides
    def _sampling_rates(self) -> Iterable[float]:
        sampling_rates = super()._sampling_rates()
        return sampling_rates ** self.power


@TaskSwitcher.register('exp_proportional_sampler')
@TaskSwitcher.register('exp_sampler')
@TaskSwitcher.register('softmax_proportional_sampler')
@TaskSwitcher.register('softmax_sampler')
class SoftmaxProportionalTaskSampler(ProportionalTaskSampler):

    def __init__(self,
                 temperature: float = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    @overrides
    def _sampling_rates(self) -> Iterable[float]:
        sampling_rates = super()._sampling_rates()
        return np.exp(sampling_rates / self.temperature)
