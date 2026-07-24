import os, argparse, time, random, string, sys, yaml, subprocess, shutil, datetime, copy
import itertools
import re
import numpy as np

import pandas as pd
from skopt import Optimizer
import json

sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

print("Running run_hyperparameter_search.py...")

from config.constants import Constants
from util.helpers import hash_from_dict

# sometimes we want to try different combinations of settings systematically (e.g. task / target combinations)
# instead of creating all combinations as different config files, we can specify a list in one config file
# and create them here on the spot
# Since the number of combinations can grow quickly, we use bayesian hyperparameter optimization to iterature through
# different combinations in a more efficient way than grid search or random search

constants = Constants()

arg_parser = argparse.ArgumentParser(description="Read in configuration")
arg_parser.add_argument("--config", help="config file", required=True)
arg_parser.add_argument("--name", help="run name", required=True)
arg_parser.add_argument("--results_base_dir", help="base directory to write results to", required=True)
arg_parser.add_argument("--parallel", help="identifier which multiple parallel runs can use to avoid rerunning the same configuration", required=False)
args = arg_parser.parse_args()

f = open(os.path.join(os.getcwd(), args.config), 'r')
RUN_PARAMETERS = yaml.load(f, Loader=yaml.FullLoader)

if args.results_base_dir is not None:
    RESULTS_BASE_DIR = args.results_base_dir  # + "_" + time.strftime("%Y-%m-%d_%H%M")
else:
    RESULTS_BASE_DIR = ""

try:
    NAME = args.name
except KeyError:
    raise ValueError("Required config for name")


def load_result(full_results_dir):
    scores_file = os.path.join(full_results_dir, "scores.json")
    if not os.path.exists(scores_file):
        log(f"Scores file {scores_file} does not exist, returning 1 penalty")
        return 1
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    try:
        mean_squared_errors = scores['scores_test']['mean_squared_error']
        mean_mse = sum(mean_squared_errors) / len(mean_squared_errors)
        return mean_mse
    except KeyError:
        log(f"Scores file {scores_file} does not contain mean_squared_error, returning penalty ")
        return 1

def get_full_results_base_dir():
    year = datetime.datetime.today().isocalendar().year
    calendar_week = datetime.datetime.today().isocalendar().week
    year_week = f"{year}_kw{calendar_week}"
    d = os.path.join(constants.RESULTS_ROOT, "runs_luha", year_week, RESULTS_BASE_DIR)
    os.makedirs(d, exist_ok=True)
    return d

def log(msg):
    print(msg)
    with open(os.path.join(get_full_results_base_dir(), "log.txt"), 'a') as f:
        f.write(str(msg) + "\n")

def make_yaml_safe(obj):
    if isinstance(obj, (np.generic,)):   # all numpy scalar types
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_yaml_safe(v) for v in obj]
    return obj


def copy_config_file():
    config_file = os.path.join(args.config)
    assert os.path.isfile(config_file), f"Config file does not exist: {config_file}"
    config_file_name = os.path.basename(config_file)
    destination_path = os.path.join(get_full_results_base_dir(), config_file_name)
    shutil.copy2(config_file, destination_path)


def run_with_params(params):
    new_run_parameters = copy.deepcopy(RUN_PARAMETERS)
    for name, p in params.items():
        new_run_parameters['config_model'][name] = make_yaml_safe(p)
    config_base_filename = os.path.splitext(os.path.basename(args.config))[0]
    config_hash = hash_from_dict({'base': args.config, 'params': params}, hash_len=8)
    beautify_val = lambda val: f"{val:.2e}" if isinstance(val, float) and (val < 1e-3 or val > 1e3) else f"{val:.3f}" if isinstance(val, float) else str(val)
    params_str = "_".join([f"{beautify_val(p)}" for name, p in params.items() if p is not None])
    yaml_file = os.path.join(config_cache_dir, f"{config_base_filename}_{params_str}_{config_hash}.yaml")
    new_name = f"{args.name}_{params_str}"

    # remove bayesian_optimization_log from new config (only relevant for bayesian optimization script, not individual runs)
    if 'bayesian_optimization_log' in new_run_parameters:
        del new_run_parameters['bayesian_optimization_log']

    with open(yaml_file, 'w') as outfile:
        yaml.dump(new_run_parameters, outfile)
        log("Writing to {}".format(yaml_file))

    # prepare command line arguments to pass to python script
    arguments = {key: args.__dict__[key] for key in args.__dict__ if key not in ['config', 'name', 'random', 'parallel']}
    python_commandline_arguments = ' '.join(["--{} {}".format(k, arguments[k]) for k in arguments if arguments[k] is not None])

    full_results_base_dir = get_full_results_base_dir()
    results_subdir = time.strftime("%Y%m%d_%H%M") + "_" + new_name + '_' + ''.join(random.choices(string.ascii_lowercase, k=4))
    full_results_dir = os.path.join(full_results_base_dir, results_subdir)
    os.makedirs(full_results_dir, exist_ok=True)

    with open(os.path.join(full_results_dir, "hyperparameters.txt"), 'w') as outfile:
        yaml.dump(params, outfile)

    command = f"python -u run.py --name {new_name} --config {yaml_file} {python_commandline_arguments} --results_dir {results_subdir}"
    log(f"Running: {command}")
    os.system(command)

    # read results from the call
    result = load_result(full_results_dir)
    return full_results_base_dir, result


def load_previous_iterations(log_file, bayesian_optimizer, print_log=0, allow_missing_file=False):
    if os.path.exists(log_file):
        if print_log > 0:
            log(f"Loading previous bayesian optimization log from {log_file}")
        previous_iterations = pd.read_csv(log_file)
        for i, row in previous_iterations.iterrows():
            param_values = [row[name] for name in params.keys()]
            assert len(param_values) == len(params), f"Invalid number of parameters in previous log: {param_values} vs {params}"
            mse = row['mse']
            try:
                bayesian_optimizer.tell(param_values, mse)
            except ValueError as e:
                # likely due to point not being within the bounds (e.g. due to changes in the search space)
                log(f"Error loading previous iteration {i} with params {param_values} and mse {mse}: {e}")
        if print_log > 0:
            log(f"Loaded {len(previous_iterations)} previous iterations into bayesian optimizer")
        if print_log > 1:
            log(f"Current best parameters: {bayesian_optimizer.Xi[np.argmin(bayesian_optimizer.yi)]} with MSE {min(bayesian_optimizer.yi)}")
            log(f"All previous errors: {bayesian_optimizer.yi}")
    else:
        if allow_missing_file:
            if print_log:
                log(f"No previous bayesian optimization log found at {log_file}, continuing without loading previous iterations")
        else:
            raise ValueError(f"No previous bayesian optimization log found at {log_file}")

    return bayesian_optimizer


def ask_with_constraints(opt, param_names):
    """
    Ask next point to evaluate, with constraints that learning_rate_lora > learning_rate
    The logic with y_lie etc. is taken from the .ask() implementation with n_points>1
    """
    opt_copy = opt.copy(random_state=opt.rng.randint(0, np.iinfo(np.int32).max))

    # find param index for learning_rate_lora and learning_rate
    if 'learning_rate' in param_names and 'learning_rate_lora' in param_names:
        learning_rate_idx = param_names.index('learning_rate')
        learning_rate_lora_idx = param_names.index('learning_rate_lora')
    else:
        return opt.ask()

    for i in range(100):
        x = opt_copy.ask()
        print(x)
        if x[learning_rate_lora_idx] > x[learning_rate_idx]:
            return x
        y_lie = np.min(opt_copy.yi) if opt_copy.yi else 0.0  # CL-min lie
        opt_copy._tell(x, y_lie)

    return x

config_cache_dir = os.path.join(constants.CACHE_DIR, "config_yaml")
os.makedirs(config_cache_dir, exist_ok=True)


copy_config_file()


hyperparameter_testing_parameters = ['learning_rate', 'learning_rate_lora', 'lora_r', 'lora_modules', 'lora_dropout',
                                     'max_audio_length_seconds', 'classifier_proj_size', 'pooler_hidden_size', 'use_lora',
                                     'classifier_layers', 'cls_dropout', 'pooler_dropout', 'hidden_dropout_prob']
params = {}
for p in hyperparameter_testing_parameters:
    try:
        params[p] = RUN_PARAMETERS['config_model'][p]
        if not isinstance(params[p], list):
            params[p] = None  # no list -> not a hyperparameter to optimize, just use the value
    except (AttributeError, KeyError):
        params[p] = None


# fix scientific notation issue in yaml loader
def fix_scientific_notation(param):
    if isinstance(param, list):
        return [fix_scientific_notation(val) for val in param]
    if isinstance(param, tuple):
        return tuple([fix_scientific_notation(val) for val in param])
    elif re.match(r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$', str(param)):
        try:
            return float(param)
        except:
            return param
    else:
        return param
params = {name: fix_scientific_notation(p) for name, p in params.items() if p is not None}
log(f"Hyperparameter config: {params}")
params_values = [p for p in params.values()]
bayesian_optimizer = Optimizer(
    dimensions=params_values,
)
log(f"Optimizing hyperparameters: {bayesian_optimizer.space}")

if 'bayesian_optimization_log' in RUN_PARAMETERS and RUN_PARAMETERS['bayesian_optimization_log'] is not None:
    assert args.parallel is None, "Cannot specify --parallel when using --bayesian_optimization_log"
    log_file = RUN_PARAMETERS['bayesian_optimization_log']
    shutil.copy2(log_file, os.path.join(get_full_results_base_dir(), "bayesian_optimization_log_start.csv"))
    bayesian_optimizer = load_previous_iterations(log_file, bayesian_optimizer, print_log=2)

iterations = []
for iteration in range(100):
    if args.parallel is not None:
        log_file = os.path.join(constants.CACHE_DIR, f"bayesian_optimization_{args.parallel}.csv")
        bayesian_optimizer = Optimizer(
            dimensions=params_values,
        )
        bayesian_optimizer = load_previous_iterations(log_file, bayesian_optimizer, print_log=1, allow_missing_file=True)

    #next_parameters_to_test = bayesian_optimizer.ask()
    next_parameters_to_test = ask_with_constraints(bayesian_optimizer, list(params.keys()))
    next_parameters_to_test_dict = {name: make_yaml_safe(next_parameters_to_test[i]) for i, name in enumerate(params.keys())}
    log("Testing parameters: " + str(next_parameters_to_test_dict) + f"[{next_parameters_to_test}]")
    full_results_base_dir, results_for_parameters = run_with_params(next_parameters_to_test_dict)
    bayesian_optimizer.tell(next_parameters_to_test, results_for_parameters)
    new_iteration = {'iteration': iteration, 'mse': results_for_parameters, **next_parameters_to_test_dict}
    iterations.append(new_iteration)
    log(f"Iteration {iteration+1}: params={next_parameters_to_test_dict}, MSE={results_for_parameters}")
    pd.DataFrame(iterations).to_csv(os.path.join(full_results_base_dir, f"hyperparameter_search_iterations.csv"), index=False)

    if args.parallel is not None:
        # append to parallel log file, so other parallel runs can pick it up (note that can be new iterations in between)
        new_iteration_df = pd.DataFrame([{**new_iteration, 'full_results_base_dir': full_results_base_dir}])
        if os.path.exists(log_file):
            log_combined = pd.concat([pd.read_csv(log_file), new_iteration_df], ignore_index=True)
        else:
            log_combined = new_iteration_df
        log_combined.to_csv(log_file, index=False)

