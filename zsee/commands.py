import itertools
import json
import logging
import argparse
import os
import random

from allennlp.common.checks import ConfigurationError
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

        subparser.add_argument('-n', '--num-runs',
                               default=1,
                               type=int,
                               help='number of runs per config')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.add_argument('--random',
                               action='store_true',
                               default=False,
                               help='random order of runs')

        subparser.add_argument('--cache-directory',
                               type=str,
                               default='',
                               help='Location to store cache of data preprocessing')

        subparser.add_argument('--cache-prefix',
                               type=str,
                               default='',
                               help='Prefix to use for data caching, giving current parameter '
                               'settings a name in the cache, instead of computing a hash')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=None,
                                 help='id of GPU to use (if not the default one from the config)')

        subparser.set_defaults(func=train_model_from_args)

        return subparser

def generate_configs(options: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = options.keys()
    base = {key: options[key][0] for key in keys}
    possible_changes = {key: options[key][1:] for key in keys}

    for num_changes in range(len(options)):
        for keys_to_change in itertools.combinations(keys, num_changes):
            values_to_pick_changes_from = [possible_changes[key]
                                           for key in keys_to_change]
            for picked_values in itertools.product(*values_to_pick_changes_from):
                config = base.copy()
                config_name = '_'
                for key_to_change, picked_value in zip(keys_to_change, picked_values):
                    config[key_to_change] = picked_value
                    config_name += f'{key_to_change}={picked_value}_'

                yield config_name, config



def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(parameter_filename=args.param_path,
                          options_filename=args.options_path,
                          serialization_dir=args.serialization_dir,
                          overrides=args.overrides,
                          file_friendly_logging=args.file_friendly_logging,
                          cache_directory=args.cache_directory,
                          cache_prefix=args.cache_prefix,
                          random=args.random,
                          num_runs=args.num_runs,
                          cuda_device=args.cuda_device)


def train_model_from_file(parameter_filename: str,
                          options_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          cache_directory: str = None,
                          cache_prefix: str = None,
                          random: bool = False,
                          num_runs: int = 1,
                          cuda_device: int = None) -> Model:
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
    num_runs : ``int``, optional (default=1)
        Number of runs per configuration.
    cuda_device: ``int``, optional (default=None)
        Cuda device id to override original config with.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    with open(options_filename, 'r', encoding='utf-8') as f:
        options = json.load(f)

    if random:
        raise NotImplementedError

    for config_name, config in generate_configs(options):
        params = OptionsParams.from_file(parameter_filename, overrides, config)

        if cuda_device is not None:
            params.params['trainer']['cuda_device'] = cuda_device

        for run_idx in range(num_runs):
            run_dir = f'{serialization_dir}/{config_name}/{run_idx}'
            try:
                # Directory level lock for multi-process training:
                #   If could take the lock, then start training
                #   If not, then someone (at least once) took the lock
                # There's no lock release.
                os.makedirs(run_dir)
            except FileExistsError:
                logger.warning('The run is already launched. Skipping...')
                continue

            train_model(params,
                        serialization_dir=run_dir,
                        file_friendly_logging=file_friendly_logging,
                        cache_directory=cache_directory,
                        cache_prefix=cache_prefix)
