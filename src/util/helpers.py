import hashlib
import hmac
import pickle
import os
import shutil
import time
import warnings

# ignore specific FutureWarning from google api core about python version
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.api_core\._python_version_support",
    message=r".*non-supported Python version.*"
)

import pandas as pd
from google.cloud.speech_v2.types.cloud_speech import AutoDetectDecodingConfig
import json
import urllib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import Tensor
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt
import types

from config.constants import Constants


def hash_from_dict(config: dict, hash_len=None):
    """
    Create hexadecimal hash of length len from a dictionary d
    Can be used to create directory for storing temporary results etc.
    """
    hash = hashlib.sha1(bytes(pickle.dumps(config))).hexdigest()
    if hash_len is not None:
        assert 0 < hash_len < len(hash)
        hash = hash[:hash_len]
    return hash

def hash_list(data: list, hash_len=None):
    dict_from_list = {'data': data}
    return hash_from_dict(dict_from_list, hash_len=hash_len)

def create_directory(path, empty_dir = False):
    """ Creates directory if it doesnt exist yet, optionally deleting all files in there """
    if not os.path.exists(path):
        os.makedirs(path)

    if empty_dir:
        shutil.rmtree(path)
        os.makedirs(path)

    try:
        os.chmod(path, 0o777)
    except PermissionError:
        pass


def python_to_json(d):
    # dump python data to json with some special handling
    # this is used to write to easily handleable text files
    def process(data):
        if isinstance(data, dict):
            return {key: process(data[key]) for key in data}  # recursive
        if isinstance(data, list):
            return [process(list_item) for list_item in data]  # recursive
        if isinstance(data, tuple):
            return tuple([process(list_item) for list_item in data])  # recursive
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return process(data.to_dict())
        elif isinstance(data, AutoDetectDecodingConfig):
            return str(data)
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        else:
            return data

    return json.dumps(process(d))

def dataset_name_to_url_part(name: str):
    """
    Make database name good for part of a url (e.g. directory name)
    """
    return urllib.parse.quote(name.replace(" ", "_").replace("(", "").replace(")", ""))



def store_obj_to_disk(obj_name, obj, base_path):
    # store obj to file, depending on type
    if isinstance(obj, np.ndarray):
        file_path = f"{obj_name}.npy"
        obj_type = "numpy"
        with open(os.path.join(base_path, file_path), 'wb') as f:
            np.save(f, obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        file_path = f"{obj_name}.pkl"
        obj_type = "pandas"
        obj.to_pickle(os.path.join(base_path, file_path))
    elif isinstance(obj, str):
        file_path = f"{obj_name}.txt"
        obj_type = "text"
        with open(os.path.join(base_path, file_path), "w") as f:
            f.write(obj)
    else:
        file_path = f"{obj_name}.pkl"
        obj_type = "pickle"
        with open(os.path.join(base_path, file_path), "wb") as f:
            pickle.dump(obj, f)

    return file_path, obj_type

def get_obj_from_disk(file_path, obj_type, base_path):
    # get obj from file, which has previously been written to there using above function store_obj_to_disk
    if obj_type == "numpy":
        with open(os.path.join(base_path, file_path), 'rb') as f:
            obj = np.load(f, allow_pickle=True)
    elif obj_type == "pandas":
        obj = pd.read_pickle(os.path.join(base_path, file_path))
    elif obj_type == "text":
        with open(os.path.join(base_path, file_path), "r") as f:
            obj = f.read()
    elif obj_type == "pickle":
        with open(os.path.join(base_path, file_path), "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Invalid obj_type {obj_type}")

    return obj


def safe_divide(a, b):
    return a / b if b != 0 else 0

def objects_equal(a, b):
    """
    Check if two objects are equal. The == operator doesn't work for numpy arrays, where an exception is raised
    if the arrays are not of the same size, and a element-wise comparison is done if they are
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b


def python_to_json(d):
    # dump python data to json with some special handling
    # this is used to write to easily handleable text files
    def process(data):
        if isinstance(data, dict):
            return {key: process(data[key]) for key in data}  # recursive
        if isinstance(data, list):
            return [process(list_item) for list_item in data]  # recursive
        if isinstance(data, tuple):
            return tuple([process(list_item) for list_item in data])  # recursive
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return process(data.to_dict())
        elif isinstance(data, AutoDetectDecodingConfig):
            return str(data)
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        if isinstance(data, Tensor):
            return data.cpu().numpy().tolist()
        else:
            return data

    return json.dumps(process(d), indent=4)


def prepare_demographics(demographics, include_socioeconomic=False, include_longitudinal=False):
    # prepare the most important demographic for use in modelling (quantitative)
    mapping = {
        'gender_unified': {'f': 0, 'm': 1},
        'education_binary': {'low-education': 0, 'high-education': 1},
        'country': {'uk': 0, 'usa': 1},
    }

    if demographics is None:
        print("Attention: No demographics provided")
        return None, mapping

    important_demographic_cols = ['age', 'gender_unified', 'education_binary', 'country']
    if include_socioeconomic:
        important_demographic_cols.append('socioeconomic')
    if include_longitudinal:
        important_demographic_cols.append('longitudinal')

    df = demographics.copy()[important_demographic_cols]
    assert demographics.shape[1] >= len(important_demographic_cols), f"There are only {demographics.shape[1]} demographic variables, should be more?"

    assert set(demographics['gender_unified']) == {'f', 'm'}, f"Gender should be f/m before preprocessing, but is {set(demographics['gender_unified'])}. Have the demographics already been preprocessed?"
    assert {'high-education', 'low-education'}.issubset(set(demographics['education_binary'])), f"Education should be high-education / low-education before preprocessing, but is {set(demographics['education_binary'])}. Have the demographics already been preprocessed?"

    for col in ['gender_unified', 'education_binary', 'country']:
        df[col] = df[col].apply(lambda x: mapping[col].get(x, None))

    if np.any(df.isna().sum(axis=0) == df.shape[0]):
        # All NaN columns --> indicates that the demographics have already been preprocessed before?
        raise ValueError(f"All NaN demographic column. Has it been preprocessed already? {df.isna().sum(axis=0)}")

    return df, mapping

def prepare_stratification(data: pd.DataFrame, combine_singleton_groups=False, age_bins=4, composite_bins=4):
    source_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age', 'mean_composite_cognitive_score']
    final_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age_binned', 'mean_composite_cognitive_score_binned']
    stratification_df = data[source_stratification_cols].copy()
    stratification_df['mean_composite_cognitive_score_binned'] = pd.qcut(stratification_df['mean_composite_cognitive_score'], q=composite_bins)
    stratification_df['age_binned'] = pd.qcut(stratification_df['age'], q=age_bins)
    stratification_df = stratification_df.drop(columns=['age', 'mean_composite_cognitive_score'])
    assert set(stratification_df.columns) == set(final_stratification_cols), f"Stratification columns should be {final_stratification_cols}, but are {stratification_df.columns}"
    assert stratification_df.shape[0] == data.shape[0], "Stratification dataframe should have the same number of rows as the original data"
    assert not stratification_df.isna().any().any(), "Stratification dataframe should not contain NaN values"
    stratification_array = stratification_df.apply(lambda row: "".join(row.astype(str)), axis=1)
    if combine_singleton_groups:
        # combine groups that only have one member to avoid issues with stratified splitting (train_test_split raises an error in this case)
        value_counts = stratification_array.value_counts()
        singleton_values = value_counts[value_counts == 1].index
        for val in singleton_values:
            stratification_array = stratification_array.replace(val, "other")
        if stratification_array[stratification_array == "other"].shape[0] == 1:  # still only one member in "other", add to random other group
            random_other_value = stratification_array[stratification_array != "other"].sample(n=1, random_state=1).values[0]
            stratification_array = stratification_array.replace("other", random_other_value)
    return stratification_array

def calculate_cohens_d(d1: pd.Series, d2: pd.Series) -> float:
    # Cohen's d effect size between d1 and d2
    # source: https://stackoverflow.com/a/71875070
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / pooled_std

def mean_absolute_percentile_error(y_true, y_pred):
    # The mean absolute error, after converting to percentiles.
    # This is a more interpretable metric of regressionin a medical context
    q = np.arange(1, 101)
    target_percentiles = np.percentile(y_true, q, method="linear")
    target_transformed = np.array([np.argmin(val > target_percentiles) for val in y_true])
    target_transformed = q[target_transformed]
    prediction_transformed = np.array([np.argmin(val > target_percentiles) if val < np.max(target_percentiles) else 99 for val in y_pred])
    prediction_transformed = q[prediction_transformed]
    return sk_metrics.mean_absolute_error(target_transformed, prediction_transformed)


composite_target_to_string_mapping = {'composite_speed': 'Speed', 'composite_language': 'Language', 'composite_executive_function': 'Executive Function', 'composite_memory': 'Memory',
                                      'speed': 'Speed', 'language': 'Language', 'executive_function': 'Executive Function', 'memory': 'Memory'}



def prepend_y_tone(
    in_path: str,
    y: float,
    out_path,
    *,
    fmin: float = 500.0,
    fmax: float = 2500.0,
    tone_duration_s: float = 0.5,
    target_tone_rms_dbfs: float = -20.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
    mono_tone_per_channel: bool = True,
):
    """
    Prepend a sine tone that encodes y by its frequency, then save audio.

    Mapping:
        y -> f(y) = fmin + (fmax - fmin) * (clip(y, y_min, y_max) - y_min) / (y_max - y_min)

    Args:
        in_path: input audio path (any format supported by soundfile).
        y: standardized target (mean≈0, std≈1). Values are clipped to [y_min, y_max].
        out_path: where to save. If None, uses '<stem>__ytoned<same-ext>'.
        fmin, fmax: frequency range for encoding (Hz).
        tone_duration_s: tone length to prepend (seconds).
        target_tone_rms_dbfs: tone RMS level relative to full-scale=1.0 (e.g., -20 dBFS).
        y_min, y_max: bounds used to map standardized y to frequency.
        mono_tone_per_channel: if True, generate one mono tone and replicate across channels;
                               if False, generate independent phases per channel (same freq).
    Returns:
        out_path (str), used_frequency_hz (float)
    """
    # --- I/O with soundfile, fallback to scipy if needed ---
    try:
        import soundfile as sf
        data, sr = sf.read(in_path, always_2d=True)   # shape [num_samples, num_channels], float or int
        info = sf.info(in_path)
        subtype = getattr(info, "subtype", None)      # preserve original subtype if possible
        ext = in_path.split(".")[-1].lower()
        writer = ("soundfile", sf, subtype, ext)
    except Exception:
        # Fallback for WAV only
        from scipy.io import wavfile
        sr, data = wavfile.read(in_path)              # shape [num_samples,] or [num_samples, channels]
        if data.ndim == 1:
            data = data[:, None]
        # Convert to float32 in [-1, 1] for processing
        if np.issubdtype(data.dtype, np.integer):
            maxi = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / maxi
            original_dtype = "int"                    # remember for writing back
        else:
            data = data.astype(np.float32)
            original_dtype = "float"
        writer = ("scipy", wavfile, original_dtype, "wav")

    num_samples, num_channels = data.shape

    # --- Build the tone ---
    y_clipped = np.clip(y, y_min, y_max)
    freq = fmin + (fmax - fmin) * (y_clipped - y_min) / (y_max - y_min)

    n_tone = int(round(tone_duration_s * sr))
    t = np.arange(n_tone, dtype=np.float32) / float(sr)

    # Sine tone; same freq per channel. Phase handling:
    if mono_tone_per_channel or num_channels == 1:
        tone = np.sin(2 * np.pi * freq * t, dtype=np.float32)[:, None]
        tone = np.repeat(tone, num_channels, axis=1)
    else:
        # different random phase per channel (same freq)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2*np.pi, size=num_channels).astype(np.float32)
        tone = np.stack([np.sin(2*np.pi*freq*t + ph) for ph in phases], axis=1).astype(np.float32)

    # Short fades to avoid clicks (5 ms)
    fade_len = max(1, int(round(0.005 * sr)))
    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
    fade_out = fade_in[::-1]
    tone[:fade_len, :] *= fade_in[:, None]
    tone[-fade_len:, :] *= fade_out[:, None]

    # Scale tone to target RMS (per-channel)
    eps = 1e-12
    target_rms = 10 ** (target_tone_rms_dbfs / 20.0)  # full-scale reference = 1.0
    cur_rms = np.sqrt(np.mean(tone**2, axis=0) + eps)
    tone = tone * (target_rms / (cur_rms + eps))[None, :]

    # --- Concatenate (prepend) without touching original program audio ---
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    out = np.vstack([tone, data])

    # --- Save ---
    if out_path is None:
        import os
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}__ytoned{ext}"

    backend, mod, meta, ext = writer
    if backend == "soundfile":
        # Try to preserve original format & subtype, else default to 32-bit float
        kwargs = {}
        if meta:
            kwargs["subtype"] = meta
        mod.write(out_path, out, sr, **kwargs)
    else:
        # scipy.io.wavfile only supports integer PCM or float32 for WAV
        if meta == "int":
            wav_out = np.clip(out, -1.0, 1.0)
            wav_out = (wav_out * 32767.0).astype(np.int16)
        else:
            wav_out = out.astype(np.float32)
        mod.write(out_path, sr, wav_out)

    return out_path, float(freq)

def plot_target_prediction(target_and_prediction, target_variable, figsize=(5, 5), fig_store_path=None, csv_store_path=None, prediction_column='prediction'):
    """
    Plot the target variable against the prediction, including histograms of both variables.
    """
    assert isinstance(target_and_prediction, pd.DataFrame), "target_and_prediction should be a pandas DataFrame"
    assert target_variable in target_and_prediction.columns, f"Target variable {target_variable} not found in data"
    assert prediction_column in target_and_prediction.columns, f"Prediction column {prediction_column} not found in data"

    if fig_store_path is not None:
        assert os.path.isdir(
            os.path.dirname(fig_store_path)), f"Directory {os.path.dirname(fig_store_path)} does not exist"
        assert any([fig_store_path.endswith(ending) for ending in
                    ['.png', '.pdf']]), f"Invalid file extension for figure: {fig_store_path}. Use .png or .pdf"
    if csv_store_path is not None:
        assert os.path.isdir(
            os.path.dirname(csv_store_path)), f"Directory {os.path.dirname(csv_store_path)} does not exist"
        assert csv_store_path.endswith('.csv'), f"Invalid file extension for CSV: {csv_store_path}. Use .csv"

    # Calculate metrics
    r2 = sk_metrics.r2_score(target_and_prediction[target_variable], target_and_prediction[prediction_column])
    explained_variance = sk_metrics.explained_variance_score(target_and_prediction[target_variable],
                                                          target_and_prediction[prediction_column])
    mean_absolute_error = sk_metrics.mean_absolute_error(target_and_prediction[target_variable],
                                                      target_and_prediction[prediction_column])
    correlation = np.corrcoef(target_and_prediction[target_variable],
                              target_and_prediction[prediction_column])[0, 1]
    mse = sk_metrics.mean_squared_error(target_and_prediction[target_variable],
                                     target_and_prediction[prediction_column])

    # Create the plot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, width_ratios=(10, 1), height_ratios=(1, 10))
    ax_main, ax_histx, ax_histy = axes[1][0], axes[0][0], axes[1][1]
    axes[0][1].set_axis_off()

    min_val, max_val = target_and_prediction[[target_variable, prediction_column]].min().min(), target_and_prediction[
        [target_variable, prediction_column]].max().max()
    ax_main.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.5, label="Perfect prediction")
    ax_main.scatter(target_and_prediction[target_variable], target_and_prediction[prediction_column], alpha=0.5,
                    marker='.', s=2,
                    label=f'R²: {r2:.3f}\nExplainedVar: {explained_variance:.3f}\nMAE: {mean_absolute_error:.3f}\nMSE: {mse:.2f}\nPearsonCorr: {correlation:.3f}')
    ax_main.axvline(target_and_prediction[target_variable].mean(), c='k', alpha=0.1, linestyle="--", label="Mean")
    ax_main.axhline(target_and_prediction[prediction_column].mean(), c='k', alpha=0.1, linestyle="--")
    ax_main.set_xlabel(f'Actual Target ({target_variable})')
    ax_main.set_ylabel('Predicted Target')
    fig.suptitle(f'Prediction vs Target (n={target_and_prediction.shape[0]})')
    ax_main.legend()

    # joint histogram bins
    _, bins = pd.cut(pd.concat((target_and_prediction[target_variable], target_and_prediction[prediction_column])), bins=20,
                     retbins=True)

    # Histogram for the x-axis variable
    n, _, _ = ax_histx.hist(target_and_prediction[target_variable], bins=bins, alpha=0.5, color='grey',
                            density=True)
    ax_histx.axis('off')  # Turn off axis labels/ticks
    mean, std = np.mean(target_and_prediction[target_variable]), np.std(target_and_prediction[target_variable])
    text_position = np.max(n) / 3
    ax_histx.text(mean, text_position, f"{mean:.2f}+-{std:.2f}", ha='center', alpha=0.5)

    # Histogram for the y-axis variable
    n, _, _ = ax_histy.hist(target_and_prediction[prediction_column], bins=bins, orientation='horizontal', alpha=0.5,
                            color='grey', density=True)
    ax_histy.axis('off')  # Turn off axis labels/ticks
    mean, std = np.mean(target_and_prediction[prediction_column]), np.std(target_and_prediction[prediction_column])
    text_position = np.max(n) / 4
    ax_histy.text(text_position, mean, f"{mean:.2f}+-{std:.2f}", ha='center', rotation=-90, va='center', alpha=0.5)

    plt.subplots_adjust(wspace=0, hspace=0)

    if fig_store_path is not None:
        plt.savefig(fig_store_path)
        plt.close()
    else:
        plt.show()

    if csv_store_path is not None:
        target_and_prediction.to_csv(csv_store_path)


def get_feature_group_by_column_name(column_name: str, group_version: str = None):
    """
    Get the feature group of a feature based on its column name
    group_version == 'global': --> return global everywhere (all the same group)
    """
    if group_version == 'global':
        return 'global'
    elif group_version is None:
        if column_name.startswith('dem_'):
            return 'demographic'
        elif column_name.split('_')[0] in ['smile', 'pause', 'phon']:
            return 'acoustic'
        elif column_name.split('_')[0] in ['lit', 'sung', 'liwc', 'iu']:
            return 'linguistic'
        else:
            return None
    else:
        raise ValueError(f"Invalid group_version {group_version}")

def convert_namespace_to_dict_recursive(val):
    if isinstance(val, types.SimpleNamespace):
        return {k: convert_namespace_to_dict_recursive(v) for k, v in vars(val).items()}
    elif isinstance(val, (list, tuple)):
        return type(val)(convert_namespace_to_dict_recursive(x) for x in val)
    else:
        return val

def deep_dict_diff(d1, d2, path=""):
    assert isinstance(d1, dict) and isinstance(d2, dict)
    diffs = {}

    # All keys from both dicts
    keys = set(d1.keys()) | set(d2.keys())

    for k in keys:
        v1 = d1.get(k)
        v2 = d2.get(k)
        current_path = f"{path}.{k}" if path else k

        # Case 1: key missing in one dict
        if k not in d1:
            diffs[current_path] = {"type": "added", "new_value": v2}
        elif k not in d2:
            diffs[current_path] = {"type": "removed", "old_value": v1}

        # Case 2: both values are dicts → recurse
        elif isinstance(v1, dict) and isinstance(v2, dict):
            nested = deep_dict_diff(v1, v2, current_path)
            diffs.update(nested)

        # Case 3: values differ
        elif v1 != v2:
            diffs[current_path] = {"type": "changed", "old_value": v1, "new_value": v2}

    return diffs

def store_timing_information(start_time, out_dir, description):
    end_time = time.time()
    duration_seconds = end_time - start_time
    with open(os.path.join(out_dir, "runtime.txt"), "a") as f:
        f.write(f"Duration: {duration_seconds:.4f} seconds, Description: {description}, Endtime: {end_time}\n")
    #print(f"TIMING: {description} took {duration_seconds:.4f} seconds")

def _create_train_val_split(train_df, val_size=0.1, random_state=None):
    # train / validation split within training set, iteratively adapt age_bins and composite_bins if to avoid too many groups, depending on sample size
    assert list(train_df.index) == list(range(train_df.shape[0])), "train_df should have a default integer index"
    assert 'sample_name' in train_df.columns, "train_df should contain a sample_name column"
    for age_bins, composite_bins in [(3, 3), (2, 3), (2, 2), (1, 2), (1, 1)]:
        stratification_array = prepare_stratification(train_df, combine_singleton_groups=True, age_bins=age_bins, composite_bins=composite_bins)
        try:
            train_sample_name, val_sample_name = train_test_split(train_df.sample_name, test_size=val_size, random_state=random_state, shuffle=True, stratify=stratification_array)
            return train_sample_name, val_sample_name
        except Exception as e:
            continue

def resolve_cv_fold_assignment(regression_df, dataset, constants, cv_splits, data_split_version=None, run_parameters=None):
    """
    Centralized CV fold assignment logic shared by all regression-style models.

    Handles predefined CV files (LUHA2024 5-fold, LUHACombined 10-fold),
    on-the-fly StratifiedKFold as fallback, and train/test splits (cv_splits=1).

    Returns regression_df with 'test_split' column merged in (and optionally
    split{N}_val / train_val_split columns from predefined CV files).
    """
    from data_analysis.dataloader.dataset import DatasetType

    if cv_splits > 1:
        if cv_splits == 5 and dataset.type == DatasetType.LUHA2024:
            data_split = dataset.config['data_split']
            if data_split in ['n100', 'n400'] or os.path.exists(data_split):
                assert data_split_version is None, "data_split_version must be None when using n100 or n400 data_split or custom file"
            if data_split_version == "val20":
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_CV5_VAL20
            elif data_split_version == "resampled":
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_CV5_RESAMPLED
            elif data_split_version == "random":
                assert run_parameters is not None, "run_parameters required for random data_split_version"
                cv_split_file = os.path.join(run_parameters.results_dir, "cv_fold_assignment.csv")
                create_cv_train_val_test_folds(regression_df, cv_splits, out_file=cv_split_file)
            elif data_split == "n100":
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_CV5_N100
            elif data_split == "n400":
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_CV5_N400
            elif os.path.exists(data_split):
                cv_split_file = data_split
            else:
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_CV5
            print(f"Using predefined {cv_splits}-fold cross validation assignment from {cv_split_file}")
            cv_fold_assignment = pd.read_csv(cv_split_file)
            cv_fold_assignment['sample_name'] = cv_fold_assignment['sample_name'].astype(str)
            regression_df = regression_df.merge(cv_fold_assignment, on='sample_name', how='left')
            assert not regression_df['test_split'].isna().any(), "Some samples do not have a fold assignment"
            assert regression_df['test_split'].nunique() == cv_splits, \
                f"Number of folds in fold assignment {regression_df['test_split'].nunique()} does not match cv_splits {cv_splits}"

        elif cv_splits == 10 and dataset.type == DatasetType.LUHACombined:
            split_2024 = dataset.config.get('split_2024', 'full')
            data_split = dataset.config.get('data_split', 'full')
            only_longitudinal = dataset.config.get('only_longitudinal_participants', False)
            if split_2024 == 'train':
                assert not only_longitudinal, "not implemented for only_longitudinal_participants=True"
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_TRAIN_CV10
            elif data_split == 'wave1_cohortA':
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_WAVE1COHORTA_CV10
            else:
                if only_longitudinal:
                    cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_ONLY_LONGITUDINAL_CV10
                else:
                    cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_FULL_CV10

            print(f"Using predefined {cv_splits}-fold cross validation assignment from {cv_split_file} (split_2024={split_2024}, data_split={data_split}, only_longitudinal={only_longitudinal}, dataset.config={dataset.config})")
            cv_fold_assignment = pd.read_csv(cv_split_file)
            cv_fold_assignment['sample_name'] = cv_fold_assignment['sample_name'].astype(str)
            regression_df = regression_df.merge(cv_fold_assignment, on='sample_name', how='left')
            n_missing = regression_df['test_split'].isna().sum()
            if n_missing > 0:
                print(f"WARNING: {n_missing} samples do not have a fold assignment (likely dropped during fold creation due to missing stratification info). Dropping them.")
                regression_df = regression_df.dropna(subset=['test_split']).reset_index(drop=True)
            assert regression_df['test_split'].nunique() == cv_splits, \
                f"Number of folds in fold assignment {regression_df['test_split'].nunique()} does not match cv_splits {cv_splits}"

        elif cv_splits == 5 and dataset.type == DatasetType.LUHACombined:
            split_2024 = dataset.config.get('split_2024', 'full')
            data_split = dataset.config.get('data_split', 'full')
            only_longitudinal = dataset.config.get('only_longitudinal_participants', False)
            if split_2024 == 'train':
                assert not only_longitudinal, "not implemented for only_longitudinal_participants=True"
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_TRAIN_CV5_WITH_VAL
            elif data_split == 'wave1_cohortA':
                cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_WAVE1COHORTA_CV5_WITH_VAL
            else:
                if only_longitudinal:
                    cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_ONLY_LONGITUDINAL_CV5_WITH_VAL
                else:
                    cv_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_FULL_CV5_WITH_VAL
            print(f"Using predefined {cv_splits}-fold cross validation assignment from {cv_split_file}")
            cv_fold_assignment = pd.read_csv(cv_split_file)
            cv_fold_assignment['sample_name'] = cv_fold_assignment['sample_name'].astype(str)
            regression_df = regression_df.merge(cv_fold_assignment, on='sample_name', how='left')
            n_missing = regression_df['test_split'].isna().sum()
            if n_missing > 0:
                print(f"WARNING: {n_missing} samples do not have a fold assignment (likely dropped during fold creation due to missing stratification info). Dropping them.")
                regression_df = regression_df.dropna(subset=['test_split']).reset_index(drop=True)
            assert regression_df['test_split'].nunique() == cv_splits, \
                f"Number of folds in fold assignment {regression_df['test_split'].nunique()} does not match cv_splits {cv_splits}"


        else:
            assert data_split_version is None, \
                "Predefined CV splits are only available for LUHA2024 5-fold CV and LUHACombined 10-fold CV"
            print("No predefined CV folds, preparing CV folds on the fly...")
            regression_df['test_split'] = np.ones((regression_df.shape[0],)) * -1
            kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)
            stratification_array = prepare_stratification(regression_df)
            for split_idx, (train_indices, test_indices) in enumerate(kfold.split(regression_df, y=stratification_array)):
                assert np.all(regression_df.iloc[test_indices, regression_df.columns.get_loc('test_split')] == -1)
                regression_df.iloc[test_indices, regression_df.columns.get_loc('test_split')] = split_idx

    elif cv_splits == 1:
        if dataset.type == DatasetType.LUHA2024:
            if data_split_version == "val20":
                train_val_test_split_file = constants.TRAIN_VAL_TEST_DATASPLIT_VAL20
            elif data_split_version == "resampled":
                train_val_test_split_file = constants.TRAIN_VAL_TEST_DATASPLIT_RESAMPLED
            elif data_split_version == "random":
                assert run_parameters is not None, "run_parameters required for random data_split_version"
                train_val_test_split_file = os.path.join(run_parameters.results_dir, "train_val_test_split_file.csv")
                create_cv_train_val_folds_full(regression_df, out_file=train_val_test_split_file, constants=constants)
            else:
                train_val_test_split_file = constants.TRAIN_VAL_TEST_DATASPLIT
            train_val_test_datasplit = pd.read_csv(train_val_test_split_file).astype(str)
            regression_df = regression_df.merge(train_val_test_datasplit, on='sample_name', how='left')
            regression_df['test_split'] = np.where(regression_df['split'] == 'test', 0, -1)

        elif dataset.type == DatasetType.LUHACombined and False:
            # train on single participants (across years), test on longitudinal participants
            train_val_test_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_TRAIN_SINGLE_TEST_LONGITUDINAL
            print(f"Using predefined train/test folds: train on single participants, test on longitudinal participants, from {train_val_test_split_file}")
            train_val_test_datasplit = pd.read_csv(train_val_test_split_file).astype(str)
            regression_df = regression_df.merge(train_val_test_datasplit, on='sample_name', how='left')
            regression_df['test_split'] = np.where(regression_df['split'] == 'test', 0, -1)
        elif dataset.type == DatasetType.LUHACombined:
            # train on Wave 1 and cohort A, test on Cohort B
            train_val_test_split_file = constants.CV_FOLD_ASSIGNMENT_COMBINED_WAVE1COHORTA_TEST_COHORTB_WITH_VAL
            print(f"Using predefined train/test folds: train on wave1 / cohort A, test on cohort B, from {train_val_test_split_file}")
            train_val_test_datasplit = pd.read_csv(train_val_test_split_file).astype(str)
            regression_df = regression_df.merge(train_val_test_datasplit, on='sample_name', how='left')
            regression_df['test_split'] = np.where(regression_df['split'] == 'test', 0, -1)

        else:
            raise ValueError(f"Pre-defined CV splits with cv_splits=1 are only available for LUHA2024 and LUHACombined datasets")

    else:
        raise ValueError(f"Invalid cv_splits value {cv_splits}")

    print("CV split statistics", regression_df.test_split.value_counts(), "\n")
    return regression_df


def create_cv_train_val_test_folds(regression_df, cv_splits, val_size=0.1, random_state=None, out_file=None):
    source_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age', 'mean_composite_cognitive_score']
    regression_df = regression_df.dropna(subset=source_stratification_cols).reset_index(drop=True).copy()  # drop samples with missing stratification info
    regression_df['test_split'] = np.ones((regression_df.shape[0],)) * -1

    keep_df = regression_df.copy()
    keep_stratification_array = prepare_stratification(keep_df, combine_singleton_groups=True, age_bins=3, composite_bins=3)  #

    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for split_idx, (train_indices, test_indices) in enumerate(kfold.split(keep_df, y=keep_stratification_array)):
        assert np.all(keep_df.iloc[test_indices, keep_df.columns.get_loc('test_split')] == -1)  # make sure test samples do not overlap
        keep_df.iloc[test_indices, keep_df.columns.get_loc('test_split')] = split_idx

        # now also train / validation split within training set, adapt age_bins and composite_bins if to avoid too many groups, depending on sample size
        keep_df[f'split{split_idx}_val'] = None
        train_df = keep_df.query(f"test_split != {split_idx}").reset_index(drop=True)
        train_sample_name, val_sample_name = _create_train_val_split(train_df, val_size=val_size, random_state=random_state)
        keep_df.loc[keep_df.sample_name.isin(val_sample_name), f'split{split_idx}_val'] = True
        keep_df.loc[keep_df.sample_name.isin(train_sample_name), f'split{split_idx}_val'] = False

    keep_df[['sample_name', 'test_split']] = keep_df[['sample_name', 'test_split']].astype(int)
    cv_folds = keep_df[['sample_name', 'test_split'] + [c for c in keep_df.columns if 'split' in c]].copy()
    if out_file is not None:
        cv_folds.to_csv(out_file, index=False)
    return cv_folds

def create_cv_train_val_folds_full(regression_df, val_size=0.1, random_state=None, out_file=None, constants:Constants=None):
    # based on analyses/kw46/stable_cv_splits_full.ipynb
    datasplit_assignment = pd.read_csv(constants.TRAIN_TEST_DATASPLIT, dtype={'study_submission_id': str})
    # add sample_names without a split, because they have missing information (first 4) or missing cognitive scores (remaining)
    study_submission_ids_to_add = ['172', '488', '631', '707',
                                   '41', '43', '44', '46', '49', '50', '54', '56', '59', '61', '99', '253', '303', '1079']
    study_submission_assignments_to_add = pd.DataFrame({
        'study_submission_id': study_submission_ids_to_add,
        'split': ['' for _ in range(len(study_submission_ids_to_add))],
    })
    overlap = set(datasplit_assignment.study_submission_id).intersection(set(study_submission_assignments_to_add.study_submission_id))
    assert len(overlap) == 0, f"There already is a train/test assignment already for study_submissions {overlap}"
    datasplit_assignment = pd.concat((
        datasplit_assignment,
        study_submission_assignments_to_add
    )).reset_index(drop=True).rename(columns={'study_submission_id': 'sample_name'})
    train_df = regression_df.merge(datasplit_assignment, on='sample_name', how='left').query(f"split == 'train'").reset_index(drop=True)
    train_sample_name, val_sample_name = _create_train_val_split(train_df, val_size=val_size, random_state=random_state)
    train_val_split = pd.DataFrame({
        'sample_name': list(train_sample_name) + list(val_sample_name),
        'train_val_split': ['train' for _ in range(len(train_sample_name))] + ['val' for _ in range(len(val_sample_name))],
    })
    datasplit_assignment_extended = datasplit_assignment.merge(train_val_split, on='sample_name', how='left')
    if out_file is not None:
        datasplit_assignment_extended.to_csv(out_file, index=False)
    return datasplit_assignment_extended

def prolific_id_to_longitudinal_id(prolific_id, git_dir_path):
    secret_file = os.path.join(git_dir_path, "src/keys/prolific_longitudinal_id_seed.txt")
    with open(secret_file, "r", encoding="utf-8") as f:
        secret = f.read().strip().encode("utf-8")
    return hmac.new(secret, prolific_id.encode('utf-8'), hashlib.sha256).hexdigest()[:10]
