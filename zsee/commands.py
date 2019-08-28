import itertools
import json
import logging
import argparse
import os
import random

from allennlp.common.file_utils import cached_path
from allennlp.common.params import _environment_variables, evaluate_file, parse_overrides, with_fallback
from allennlp.models import Model
from allennlp.common import Params
from allennlp.commands import Subcommand

from typing import Dict, Any, Iterable, Iterator, List

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from allennlp.commands.train import train_model

class OptionsParams(Params):

    def __init__(self,
                 params: Dict[str, Any],
                 history: str = "",
                 loading_from_archive: bool = False,
                 files_to_archive: Dict[str, str] = None,
                 options: Dict[str, Any] = None) -> None:
        super().__init__(params, history, loading_from_archive, files_to_archive)
        self._options = options

    def to_file(self, params_file: str, preference_orders: List[List[str]] = None) -> None:
        super().to_file(params_file, preference_orders)

        if self._options is None:
            return

        dirname = os.path.dirname(params_file)
        basename = os.path.basename(params_file)

        with open(f'{dirname}/variables_{basename}', "w") as handle:
            json.dump(self._options, handle, indent=4)


    @staticmethod
    def from_file(params_file: str,
                  params_overrides: str = "",
                  ext_vars: dict = None) -> 'Params':
        """
        Load a `Params` object from a configuration file.

        Parameters
        ----------
        params_file : ``str``
            The path to the configuration file to load.
        params_overrides : ``str``, optional
            A dict of overrides that can be applied to final object.
            e.g. {"model.embedding_dim": 10}
        ext_vars : ``dict``, optional
            Our config files are Jsonnet, which allows specifying external variables
            for later substitution. Typically we substitute these using environment
            variables; however, you can also specify them here, in which case they
            take priority over environment variables.
            e.g. {"HOME_DIR": "/Users/allennlp/home"}
        """
        if ext_vars is None:
            ext_vars = {}

        options = ext_vars
        # Escape values with `json.dumps` in order to preserve type
        ext_vars = {k: str(v) for k, v in ext_vars.items()} # json.dumps

        # redirect to cache, if necessary
        params_file = cached_path(params_file)
        ext_vars = {**_environment_variables(), **ext_vars}

        file_dict = json.loads(evaluate_file(params_file, ext_vars=ext_vars))

        overrides_dict = parse_overrides(params_overrides)
        param_dict = with_fallback(preferred=overrides_dict, fallback=file_dict)

        return OptionsParams(param_dict, options=options)


class HyperParameterSearch(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train a model.')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        subparser.add_argument('options_path',
                               type=str,
                               help='path to file describing external variables and possible values to search in')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the models and their logs')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.add_argument('--cache-directory',
                               type=str,
                               default='',
                               help='Location to store cache of data preprocessing')

        subparser.add_argument('--cache-prefix',
                               type=str,
                               default='',
                               help='Prefix to use for data caching, giving current parameter '
                               'settings a name in the cache, instead of computing a hash')

        subparser.set_defaults(func=train_model_from_args)

        return subparser

def generate_configs(options: Dict[str, Iterable[Any]]) -> Iterator[Dict[str, Any]]:
    keys = options.keys()
    values = options.values()
    for chosen_values in itertools.product(*values):
        yield dict(zip(keys, chosen_values))


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.options_path,
                          args.serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.cache_directory,
                          args.cache_prefix)


def train_model_from_file(parameter_filename: str,
                          options_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          cache_directory: str = None,
                          cache_prefix: str = None,
                          shuffle: bool = False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    parameter_filename : ``str``
        A json parameter file specifying an AllenNLP experiment.
    options_filename : ``str``
        A json parameter file to choose external variables from.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    cache_directory : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    cache_prefix : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    shuffle : ``bool``, optional (default=True)
        Whether to search in hyperparameter grid in random order or preserve order of options file.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    with open(options_filename, 'r', encoding='utf-8') as f:
        options = json.load(f)

    configs = generate_configs(options)

    if shuffle:
        configs = list(configs)
        random.shuffle(configs)

    for idx, config in enumerate(configs):
        params = OptionsParams.from_file(parameter_filename, overrides, config)

        train_model(params,
                    serialization_dir=f'{serialization_dir}/{idx}',
                    file_friendly_logging=file_friendly_logging,
                    cache_directory=cache_directory,
                    cache_prefix=cache_prefix)
