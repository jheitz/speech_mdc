import sys
import socket, getpass, subprocess
import time
from datetime import timedelta
import os
os.environ["HF_HUB_DISABLE_XET"] = "1" # disable disable Xet

sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

from config.config import Config
from config.run_parameters import RunParameters
from config.constants import Constants
from data_analysis.model.regression import Regression
from data_analysis.model.DeBERTav3 import DeBERTa
from data_analysis.model.wavlm import WavLMModel
from util.helpers import create_directory, store_timing_information
from data_analysis.dataloader.dataloader import DataLoader
from data_analysis.data_transformation.linguistic_features import LinguisticFeatures
from data_analysis.data_transformation.audio_features import AudioFeatures
from data_analysis.data_transformation.demographic_features import DemographicFeatures
from data_analysis.data_transformation.task_features import TaskFeatures
from data_analysis.dataloader.dataset import DatasetType



def run(run_parameters: RunParameters, config: Config, CONSTANTS: Constants):
    print(f"Running pipeline from user {getpass.getuser()} on host {socket.gethostname()}...")
    print("Run Parameters:")
    print(run_parameters, end="\n\n")
    print("Config:")
    print(config, end="\n\n")

    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv"], encoding="utf-8")
        print("GPU Information:")
        print(result)
    except Exception as e:
        print("nvidia-smi not found or GPU not available:", e)

    try:
        debug = config.debug
    except AttributeError:
        debug = False

    try:
        dataset = config.config_data.dataset
    except AttributeError:
        dataset = "LUHA2024"

    try:
        store_dataset_after_transformations = config.config_data.store_dataset_after_transformations
    except AttributeError:
        store_dataset_after_transformations = False

    if dataset == 'LUHA' or dataset == "LUHA2024":
        dataloader = DataLoader(debug=debug, dataset_type=DatasetType.LUHA2024, config=config, run_parameters=run_parameters)
    elif dataset == 'LUHA2026':
        dataloader = DataLoader(debug=debug, dataset_type=DatasetType.LUHA2026, config=config, run_parameters=run_parameters)
    elif dataset == 'LUHACombined':
        dataloader = DataLoader(debug=debug, dataset_type=DatasetType.LUHACombined, config=config, run_parameters=run_parameters)
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    data_transformers = []
    if config.data_transformers is not None:
        for p in config.data_transformers:
            if p == "Linguistic Features":
                data_transformers.append(LinguisticFeatures(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Audio Features":
                data_transformers.append(AudioFeatures(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Demographic Features":
                data_transformers.append(DemographicFeatures(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Task Features":
                data_transformers.append(TaskFeatures(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Outlier Removal and Imputation" or p == "Feature Standardizer":
                raise ValueError(f"{p} is no longer a data transformer. Instead it is implemented as a preprocessing step. This is to avoid data leakage (should not be run on the entire dataset). Update the config accordingly.")
            else:
                raise ValueError("Invalid data transformer:", p)
            pass

    data = dataloader.load_data()
    print("Data before transformations: ", data)
    for p in data_transformers:
        print(f"Running data transformer {p}...")
        start_data_transformer = time.time()
        data = p.preprocess_dataset(data)
        store_timing_information(start_data_transformer, run_parameters.results_dir, f"Data transformer {p.name}")
    print("Data after transformations:", data)
    if store_dataset_after_transformations:
        data.store_to_disk(os.path.join(run_parameters.results_dir, "dataset_after_transformations"))

    if config.model is None:
        raise ValueError("Model not specified")
    elif config.model == 'Regression':
        ModelClass = Regression
    elif config.model == 'WavLM':
        ModelClass = WavLMModel
    elif config.model == 'DeBERTa':
        ModelClass = DeBERTa
    else:
        raise ValueError(f"Invalid model specified: {config.model}")

    model = ModelClass(config=config, run_parameters=run_parameters, constants=CONSTANTS)
    model.set_data(data)
    model.prepare_data()
    model.run()

    print("done")




if __name__ == '__main__':
    # run parameters from command line arguments
    run_parameters = RunParameters.from_command_line_args()

    # configuration based on config file
    config = Config.from_yaml(run_parameters.config)

    # constants for e.g. directory paths
    CONSTANTS = Constants(local=run_parameters.local)

    # create results file
    create_directory(run_parameters.results_dir)

    start_time = time.time()
    run(run_parameters, config, CONSTANTS)
    end_time = time.time()

    with open(os.path.join(run_parameters.results_dir, "pipeline_execution_time.txt"), 'w') as f:
        f.write(f"Pipeline execution time: {str(timedelta(seconds=end_time - start_time))} ({end_time - start_time:.2f} seconds)\n")

