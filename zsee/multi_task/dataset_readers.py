import json
from typing import Dict, Iterable, List, Any

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField


@DatasetReader.register("switch")
class SwitchDatasetReader(DatasetReader):

    def __init__(
            self,
            readers: Dict[str, DatasetReader],
            default: str = None,
            dataset_field_name: str = "dataset",
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._readers = readers
        self._default_reader = default
        self._dataset_field_name = dataset_field_name

    def _process_instance(self,
                          instance: Instance,
                          dataset: str = None):
        if dataset is None:
            dataset = self._default_reader
        instance.fields[self._dataset_field_name] = MetadataField(dataset)
        return instance

    def _read_from(self,
                   file_path: str,
                   reader: str = None,
                   name: str = None,
                   **kwargs):
        if reader is None:
            reader = self._default_reader

        if name is None:
            name = reader

        for instance in self._readers[reader].read(file_path, **kwargs):
            yield self._process_instance(instance, name)

    def _read(self, file_path) -> Iterable[Instance]:
        config = {}
        if isinstance(file_path, dict):
            config = file_path
        elif isinstance(file_path, str):
            try:
                config = json.loads(file_path)
            except json.JSONDecodeError:
                config['file_path'] = file_path
        else:
            raise NotImplementedError

        yield from self._read_from(**config)

    def text_to_instance(self, *args, **kwargs) -> Instance:  # type: ignore
        reader = self._readers[self._default_reader]
        instance = reader.text_to_instance(*args, **kwargs)
        return self._process_instance(instance)


@DatasetReader.register("concat")
class ConcatDatasetReader(DatasetReader):

    def __init__(
        self,
        dataset_reader: DatasetReader,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._dataset_reader = dataset_reader

    def _decode_config(self, file_path) -> List[Any]:

        # e.g. List of configs
        #   [
        #       "some_raw_file.txt",
        #       {
        #           "dataset": "parallel_data",
        #           "file_path": "parallel.en-de"
        #       },
        #       ...
        #   ]
        if isinstance(file_path, list):
            return file_path

        # e.g. Dict - config
        #   {
        #       "dataset": "parallel_data",
        #       "file_path": "parallel.en-de"
        #   }
        if isinstance(file_path, dict):
            return [file_path]

        # e.g.
        # "some_raw_file.txt"
        # or
        # serialized version of above
        if isinstance(file_path, str):
            try:
                config = json.loads(file_path)
            except json.JSONDecodeError:
                # "some_raw_file.txt"
                return [file_path]
            else:
                # Re-check al the cases above
                return self._decode_config(config)

        raise NotImplementedError

    def _read(self, file_path) -> Iterable[Instance]:
        config = self._decode_config(file_path)
        for file_path in config:
            # Making sure the config is compatible
            # with file_path: str interface
            if not isinstance(file_path, str):
                file_path = json.dumps(file_path)

            yield from self._dataset_reader.read(file_path)

    def text_to_instance(self, *args, **kwargs) -> Instance:  # type: ignore
        return self._dataset_reader.text_to_instance(*args, **kwargs)
