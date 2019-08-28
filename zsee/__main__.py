import os
import sys
import logging

from allennlp import commands

import zsee
from zsee.commands import HyperParameterSearch

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)


def main():
    commands.main('python -m zsee', {
        'hyperparameter-search': HyperParameterSearch()
    })

if __name__ == '__main__':
    main()
