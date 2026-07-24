import sys
sys.path.insert(0, '..') # to make the import from parent dir work
from util.logging_utils.disable_transformer_progress_bars import disable_transformer_progress_bars
disable_transformer_progress_bars()

import importlib
import os
from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters
from data_preparation.preparation_logic.data_preparator import DataPreparator

# dynamically import extra logic
extra_logic_modules = {}
for round in list(os.listdir("extra_logic")):
    if round not in ['2024', '2026']: continue
    for filepath in os.listdir(os.path.join("extra_logic", round)):
        if ".py" in filepath:
            extra_logic_name = filepath.replace(".py", "")
            extra_logic_full_name = f"extra_logic.{round}.{extra_logic_name}"
            class_name = extra_logic_name.replace("_", "").title()
            module = importlib.import_module(extra_logic_full_name)
            if hasattr(module, class_name):
                extra_logic_modules[extra_logic_name] = getattr(module, class_name)



if __name__ == '__main__':
    # run parameters from command line arguments
    run_parameters = RunParameters.from_command_line_args()

    # configuration based on config file
    config = Config.from_yaml(run_parameters.config)

    # constants for e.g. directory paths
    CONSTANTS = Constants(local=run_parameters.local)

    # potential extra logic
    try:
        extra_logic_names = config.extra_logic
    except:
        extra_logic_names = []
    extra_logic_processors = []
    for logic_name in extra_logic_names:
        if logic_name in extra_logic_modules:
            extra_logic_processors.append(extra_logic_modules[logic_name]())
        else:
            raise ValueError(f'Invalid extra logic {logic_name}')

    data_preparator = DataPreparator(run_parameters, config, CONSTANTS, extra_logic_processors)
    data_preparator.prepare_data()

    print("Data preparation complete")
