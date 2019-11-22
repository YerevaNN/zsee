import logging
from typing import Dict, List, Iterable, Set, Optional, Any

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.data import DatasetReader, DataIterator, Instance
from allennlp.training.util import _set_up_cache_files

from .dataset_mingler import DatasetMingler

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataConfigurationBase(Registrable):
    default_implementation = "default"

    def __init__(self):
        self.paths: Dict[str, str] = dict()
        self.datasets_for_training: Set[str] = set()
        self.datasets_for_vocab_creation: Set[str] = set()
        self.dataset_readers: Dict[str, DatasetReader] = dict()
        self.iterators: Dict[str, DataIterator] = dict()
        self.instances: Dict[str, Iterable[Instance]] = dict()
        self.mingler: Optional[DatasetMingler] = None


@DataConfigurationBase.register('default')
class DataConfiguration(DataConfigurationBase):
    def __init__(self,
                 paths: Dict[str, str],
                 datasets_for_training: List[str] = None,
                 datasets_for_vocab_creation: List[str] = None,
                 mingler: DatasetMingler = None,
                 dataset_reader: DatasetReader = None,
                 dataset_readers: Dict[str, DatasetReader] = None,
                 validation_dataset_reader: DatasetReader = None,
                 iterator: DataIterator = None,
                 iterators: Dict[str, DataIterator] = None,
                 validation_iterator: DataIterator = None):
        super().__init__()

        self.mingler = mingler
        # TODO add checks
        self.paths = paths

        if datasets_for_training is None:
            if 'train' not in paths:
                raise ConfigurationError('If no dataset `train` was provided please'
                                         'provide `for_training` to specify which'
                                         'datasets to use for training.')
            datasets_for_training = ['train']
        self.datasets_for_training = set(datasets_for_training)

        # TODO Make sure this works in case of multi-task setup.
        if datasets_for_vocab_creation is None:
            datasets_for_vocab_creation = paths
        self.datasets_for_vocab_creation = set(datasets_for_vocab_creation)

        self.init_dataset_readers(dataset_readers, dataset_reader, validation_dataset_reader)
        self.init_iterators(iterators, iterator, validation_iterator)
        self.init_instances()

    def init_instances(self):
        self.instances = {}
        for name, path in self.paths.items():
            logger.info(f'Initializing dataset: {name}')
            self.instances[name] = self.dataset_readers[name].read(path)

        if len(self.datasets_for_training) == 1:
            train_dataset, = self.datasets_for_training
            train_instances = self.instances[train_dataset]
        else:
            if self.mingler is None:
                raise ConfigurationError("Mingler is needed in case of multiple training datasets.")
            train_instances = self.mingler({
                name: dataset
                for name, dataset in self.instances.items()
                if name in self.datasets_for_training
            })
        self.instances['_train'] = train_instances

    def init_dataset_readers(self,
                             dataset_readers: Dict[str, DatasetReader] = None,
                             dataset_reader: DatasetReader = None,
                             validation_dataset_reader: DatasetReader = None):
        if dataset_readers is None:
            dataset_readers: Dict[str, DatasetReader] = dict()

        if dataset_reader is not None:
            default_dataset_reader = dataset_reader
        elif 'train' in dataset_readers:
            default_dataset_reader = dataset_readers['train']
        else:
            raise ConfigurationError('Train dataset reader is required.')
        dataset_readers['_train'] = default_dataset_reader

        if validation_dataset_reader is not None:
            validation_dataset_reader = validation_dataset_reader
        elif 'validation' in dataset_readers:
            validation_dataset_reader = dataset_readers['validation']
        else:
            validation_dataset_reader = default_dataset_reader
        dataset_readers['_validation'] = validation_dataset_reader

        for dataset_name in self.paths:
            if dataset_name in dataset_readers:
                continue
            if dataset_name in self.datasets_for_training:
                dataset_readers[dataset_name] = default_dataset_reader
            else:
                dataset_readers[dataset_name] = validation_dataset_reader

        self.dataset_readers = dataset_readers

    def init_iterators(self,
                       iterators: Dict[str, DataIterator] = None,
                       iterator: DataIterator = None,
                       validation_iterator: DataIterator = None):
        if iterators is None:
            iterators: Dict[str, DataIterator] = dict()

        if iterator is not None:
            default_iterator = iterator
        elif 'train' in iterators:
            default_iterator = iterators['train']
            # TODO find default
            # TODO make sure `train` is in trainable datasets if it's present.
        else:
            raise ConfigurationError('Train iterator is required.')
        iterators['_train'] = default_iterator

        if validation_iterator is not None:
            validation_iterator = validation_iterator
        elif 'validation' in iterators:
            validation_iterator = iterators['validation']
        else:
            validation_iterator = default_iterator
        iterators['_validation'] = validation_iterator

        for dataset_name in self.paths:
            if dataset_name in iterators:
                continue
            if dataset_name in self.datasets_for_training:
                iterators[dataset_name] = default_iterator
            else:
                iterators[dataset_name] = validation_iterator

        self.iterators = iterators

    @classmethod
    def dataset_reader_from_params(cls,
                                   params: Params,
                                   cache_directory: str = None,
                                   cache_prefix: str = None,
                                   **extras):
        cache_dir, _ = _set_up_cache_files(params,
                                           cache_directory=cache_directory,
                                           cache_prefix=cache_prefix)
        dataset_reader = DatasetReader.from_params(params, **extras)
        if cache_dir is not None:
            dataset_reader.cache_data(cache_dir)
        return dataset_reader

    @classmethod
    def from_params(cls,
                    params: Params,
                    cache_directory: str = None,
                    cache_prefix: str = None,
                    **extras):
        datasets_extras: Dict[str, Any] = dict()

        dataset_reader_params = params.pop('dataset_reader', None)
        if dataset_reader_params is not None:
            dataset_reader = cls.dataset_reader_from_params(dataset_reader_params,
                                                            cache_directory=cache_directory,
                                                            cache_prefix=cache_prefix,
                                                            **extras)
            datasets_extras['dataset_reader'] = dataset_reader

        dataset_reader_params = params.pop('validation_dataset_reader', None)
        if dataset_reader_params is not None:
            dataset_reader = cls.dataset_reader_from_params(dataset_reader_params,
                                                            cache_directory=cache_directory,
                                                            cache_prefix=cache_prefix,
                                                            **extras)
            datasets_extras['validation_dataset_reader'] = dataset_reader

        dataset_readers_params = params.pop('dataset_readers', None)
        if dataset_readers_params is not None:
            datasets_extras['dataset_readers']: Dict[str, DatasetReader] = dict()
            for key, dataset_reader_params in dataset_readers_params.items():
                dataset_reader = cls.dataset_reader_from_params(dataset_reader_params,
                                                                cache_directory=cache_directory,
                                                                cache_prefix=cache_prefix,
                                                                **extras)
                datasets_extras['dataset_readers'][key] = dataset_reader

        return super().from_params(params, **datasets_extras, **extras)

