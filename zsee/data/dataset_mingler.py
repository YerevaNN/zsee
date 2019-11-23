from typing import Dict, Iterable, Set

from allennlp.common import Registrable
from allennlp.data import Instance
from allennlp.data.fields import MetadataField


class DatasetMingler(Registrable):
    """
    Our ``DataIterator`` class expects a single dataset;
    this is an abstract class for combining multiple datasets into one.

    You could imagine an alternate design where there is a
    ``MinglingDatasetReader`` that wraps multiple dataset readers,
    but then somehow you'd have to get it multiple file paths.
    """

    def __init__(self, lazy: bool = False):
        self.lazy = lazy

    def __call__(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        if len(datasets) == 1:
            dataset, = datasets.values()
            return dataset

        iterable = self.mingle(datasets)
        if not self.lazy:
            iterable = list(iterable)
        return iterable

    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        raise NotImplementedError


@DatasetMingler.register("round-robin")
class RoundRobinMingler(DatasetMingler):
    """
    Cycle through datasets, ``take_at_time`` instances at a time.
    """
    def __init__(self,
                 dataset_name_field: str = "dataset",
                 take_at_a_time: int = 1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.dataset_name_field = dataset_name_field
        self.take_at_a_time = take_at_a_time

    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        iterators = {name: iter(dataset) for name, dataset in datasets.items()}
        done: Set[str] = set()

        while iterators.keys() != done:
            for name, iterator in iterators.items():
                if name in done:
                    continue
                try:
                    for _ in range(self.take_at_a_time):
                        instance = next(iterator)
                        instance.fields[self.dataset_name_field] = MetadataField(name)
                        yield instance
                except StopIteration:
                    done.add(name)
