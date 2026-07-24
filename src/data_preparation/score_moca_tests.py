import sys
sys.path.insert(0, '..') # to make the import from parent dir work
from util.logging_utils.disable_transformer_progress_bars import disable_transformer_progress_bars
disable_transformer_progress_bars()

from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters

from test_scoring.moca_test_scorer import MoCATestScorer



if __name__ == '__main__':
    # run parameters from command line arguments
    run_parameters = RunParameters.from_command_line_args()

    # configuration based on config file
    config = Config.from_yaml(run_parameters.config)

    # constants for e.g. directory paths
    CONSTANTS = Constants(local=run_parameters.local)

    moca_test_scorer = MoCATestScorer(run_parameters, config, CONSTANTS)
    moca_test_scorer.score_tests()

    print("MoCA test scoring complete")
