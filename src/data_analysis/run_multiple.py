import os, argparse, time, random, string, sys, yaml, subprocess, shutil, datetime, copy
import itertools

sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

print("Running run_multiple.py...")

from config.constants import Constants
from util.helpers import hash_from_dict

# sometimes we want to try different combinations of settings systematically (e.g. task / target combinations)
# instead of creating all combinations as different config files, we can specify a list in one config file
# and create them here on the spot

constants = Constants()

arg_parser = argparse.ArgumentParser(description="Read in configuration")
arg_parser.add_argument("--config", help="config file", required=True)
arg_parser.add_argument("--name", help="run name", required=True)
arg_parser.add_argument("--results_base_dir", help="base directory to write results to", required=True)
arg_parser.add_argument("--random", help="randomly iterate through parameters", required=False)
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


try:
    tasks = RUN_PARAMETERS['config_data']['task']
    if not isinstance(tasks, list):
        tasks = [tasks]
except:
    raise ValueError("A task should be specified in the config file")

try:
    targets = RUN_PARAMETERS['config_model']['target_variable']
    if not isinstance(targets, list):
        targets = [targets]
except:
    raise ValueError("A target variable should be specified in the config file")

try:
    svr_parameters = RUN_PARAMETERS['config_model']['svr_parameters']
    if not isinstance(svr_parameters, list):
        svr_parameters = [svr_parameters]
except:
    svr_parameters = [None]


try:
    learning_rates = RUN_PARAMETERS['config_model']['learning_rate']
    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates]
except:
    learning_rates = [None]

try:
    learning_rates_lora = RUN_PARAMETERS['config_model']['learning_rate_lora']
    if not isinstance(learning_rates_lora, list):
        learning_rates_lora = [learning_rates_lora]
except:
    learning_rates_lora = [None]


config_cache_dir = os.path.join(constants.CACHE_DIR, "config_yaml")
os.makedirs(config_cache_dir, exist_ok=True)

parameters_to_iterate = [tasks, targets, learning_rates, learning_rates_lora, svr_parameters]
parameter_sets = list(itertools.product(*parameters_to_iterate))
if args.random:
    random.shuffle(parameter_sets)
print(f"Running {len(parameter_sets)} configurations: ")
print(*parameter_sets, sep="\n")
print("\n\n")

def dict_to_str(d):
    if isinstance(d, dict):
        return "-".join([f"{k}{v}" for k, v in d.items()])
    else:
        return str(d)

for params in parameter_sets:
    task, target, learning_rate, learning_rate_lora, svr_parameters = params
    new_run_parameters = copy.deepcopy(RUN_PARAMETERS)
    new_run_parameters['config_data']['task'] = task
    new_run_parameters['config_model']['target_variable'] = target
    new_run_parameters['config_model']['learning_rate'] = learning_rate
    new_run_parameters['config_model']['learning_rate_lora'] = learning_rate_lora
    new_run_parameters['config_model']['svr_parameters'] = svr_parameters
    config_base_filename = os.path.splitext(os.path.basename(args.config))[0]
    config_hash = hash_from_dict({'base': args.config, 'params': params}, hash_len=6)
    params_str = "_".join([f"{dict_to_str(p)}" for p in params if p is not None])
    yaml_file = os.path.join(config_cache_dir, f"{config_base_filename}_{params_str}_{config_hash}.yaml")
    new_name = f"{args.name}_{params_str}"

    # if args.parallel is given, we check if the config yaml has already been created, indicating that another process has already run this configuration, in which case we skip
    if args.parallel is not None:
        parallel_hash = f"p{args.parallel}"
        yaml_file = os.path.join(config_cache_dir, f"{config_base_filename}_{params_str}_{parallel_hash}_{config_hash}.yaml")
        if os.path.isfile(yaml_file):
            print(f"Skipping already run configuration {yaml_file} because --parallel {args.parallel} was given")
            continue

    with open(yaml_file, 'w') as outfile:
        yaml.dump(new_run_parameters, outfile)
        print("Writing to {}".format(yaml_file))

    # prepare command line arguments to pass to python script
    arguments = {key: args.__dict__[key] for key in args.__dict__ if key not in ['config', 'name', 'random', 'parallel']}
    python_commandline_arguments = ' '.join(["--{} {}".format(k, arguments[k]) for k in arguments if arguments[k] is not None])

    command = f"python -u run.py --name {new_name} --config {yaml_file} {python_commandline_arguments} "
    print(f"Running: {command}")
    os.system(command)

