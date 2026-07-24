#  based on /analyses/kw46/multimodal_language_different_versions_v6_no_leakage.ipynb




import sys
import socket, getpass, subprocess
import time
from datetime import timedelta
import os
import pickle
from typing import NamedTuple

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(sys.path[0] + '/../..')  # to make the import from parent dir util work

from config.config import Config
from config.run_parameters import RunParameters
from config.constants import Constants
from util.helpers import create_directory, python_to_json

from util.generalized_ridge import GeneralizedRidge

import pandas as pd
import numpy as np
import json
from scipy.stats import bootstrap
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from datetime import datetime


class DataSplit(NamedTuple):
    X: pd.DataFrame
    y: pd.Series
    sample_names: pd.Series
    tasks: pd.Series

class Data(NamedTuple):
    train: DataSplit
    val: DataSplit
    test: DataSplit

    def select_features(self, features):
        return Data(
            train=DataSplit(
                X=self.train.X[features].copy(),
                y=self.train.y,
                sample_names=self.train.sample_names,
                tasks=self.train.tasks,
            ),
            val=DataSplit(
                X=self.val.X[features].copy(),
                y=self.val.y,
                sample_names=self.val.sample_names,
                tasks=self.val.tasks,
        ),
            test=DataSplit(
                X=self.test.X[features].copy(),
                y=self.test.y,
                sample_names=self.test.sample_names,
                tasks=self.test.tasks,
            )
        )

class CombineWavlmDeberta:
    def __init__(self, run_parameters: RunParameters, config: Config, constants: Constants):
        self.run_parameters = run_parameters
        self.config = config
        self.CONSTANTS = constants

        try:
            self.debug = config.debug
        except AttributeError:
            self.debug = False

        try:
            self.n_runs = config.n_runs
        except AttributeError:
            self.n_runs = 500

        try:
            self.target_variable = config.target_variable
        except AttributeError:
            self.target_variable = None
        assert self.target_variable is not None, "target_variable must be specified in config for CombineWavlmDeberta"

        try:
            self.results_dir_audio = config.results_dir_audio
        except AttributeError:
            self.results_dir_audio = None
        assert self.results_dir_audio is not None, "results_dir_audio must be specified in config for CombineWavlmDeberta"
        assert os.path.isdir(self.results_dir_audio), f"results_dir_audio does not exist: {self.results_dir_audio}"

        try:
            self.results_dir_text = config.results_dir_text
        except AttributeError:
            self.results_dir_text = None
        assert self.results_dir_text is not None, "results_dir_text must be specified in config for CombineWavlmDeberta"
        assert os.path.isdir(self.results_dir_text), f"results_dir_text does not exist: {self.results_dir_text}"

        try:
            self.full_results_dir_audio = config.full_results_dir_audio
        except AttributeError:
            self.full_results_dir_audio = None
        if self.full_results_dir_audio is not None:
            assert os.path.isdir(self.full_results_dir_audio), f"full_results_dir_audio does not exist: {self.full_results_dir_audio}"

        try:
            self.full_results_dir_text = config.full_results_dir_text
        except AttributeError:
            self.full_results_dir_text = None
        if self.full_results_dir_text is not None:
            assert os.path.isdir(self.full_results_dir_text), f"full_results_dir_text does not exist: {self.full_results_dir_text}"

        self.has_full_data = not (self.full_results_dir_audio is None or self.full_results_dir_text is None)

        try:
            self.version = config.version
        except AttributeError:
            self.version = 'v6'
        assert self.version in ['v6', 'v8_global_alpha', 'v9_fixed_alpha']

        print(f"Initializing CombineWavlmDeberta (version {self.version})")


    def _load_data(self, base_results_dir, split, postfix=None):
        test_data = pd.read_csv(os.path.join(base_results_dir, f'test_feature_values_split{split}.csv'))
        test_df = pd.read_csv(os.path.join(base_results_dir, f'test_df_split{split}.csv'))
        assert np.all(np.abs(test_data['targets'] - test_df[self.target_variable]) < 1e-5), \
            f"Targets do not match for test data in split {split} (base dir {base_results_dir}). Is the target variable in the config correct?"
        test_data['sample_name'] = test_df['sample_name']
        test_data['task'] = test_df['task']

        train_data = pd.read_csv(os.path.join(base_results_dir, f'train_feature_values_split{split}.csv'))
        train_df = pd.read_csv(os.path.join(base_results_dir, f'train_df_split{split}.csv'))
        assert np.all(np.abs(train_data['targets'] - train_df[self.target_variable]) < 1e-5)
        train_data['sample_name'] = train_df['sample_name']
        train_data['task'] = train_df['task']

        val_data = pd.read_csv(os.path.join(base_results_dir, f'val_feature_values_split{split}.csv'))
        val_df = pd.read_csv(os.path.join(base_results_dir, f'val_df_split{split}.csv'))
        val_data['sample_name'] = val_df['sample_name']
        val_data['task'] = val_df['task']

        if postfix is not None:
            test_data = test_data.rename(
                columns={c: f"{c}_{postfix}" for c in test_data.columns if c not in ['sample_name', 'task']})
            train_data = train_data.rename(
                columns={c: f"{c}_{postfix}" for c in train_data.columns if c not in ['sample_name', 'task']})
            val_data = val_data.rename(
                columns={c: f"{c}_{postfix}" for c in val_data.columns if c not in ['sample_name', 'task']})

        return train_data, test_data, val_data

    def _combine_datasets(self, data1, data2):
        smaller_size = min(len(data1), len(data2))
        larger_size = max(len(data1), len(data2))
        with open(os.path.join(self.run_parameters.results_dir, "intersection.txt"), 'w') as f:
            f.write(str(set(data1[['sample_name', 'task']].itertuples(index=False, name=None)).intersection(set(data2[['sample_name', 'task']].itertuples(index=False, name=None)))))
        with open(os.path.join(self.run_parameters.results_dir, "union.txt"), 'w') as f:
            f.write(str(set(data1[['sample_name', 'task']].itertuples(index=False, name=None)).union(set(data2[['sample_name', 'task']].itertuples(index=False, name=None)))))
        assert len(set(data1[['sample_name', 'task']].itertuples(index=False, name=None)).intersection(set(data2[['sample_name', 'task']].itertuples(index=False, name=None)))) == smaller_size, f"{len(set(data1[['sample_name', 'task']].itertuples(index=False, name=None)).intersection(set(data2[['sample_name', 'task']].itertuples(index=False, name=None))))} vs.{smaller_size}"
        assert len(set(data1[['sample_name', 'task']].itertuples(index=False, name=None)).union(set(data2[['sample_name', 'task']].itertuples(index=False, name=None)))) == larger_size, f"{len(set(data1[['sample_name', 'task']].itertuples(index=False, name=None)).union(set(data2[['sample_name', 'task']].itertuples(index=False, name=None))))} vs.{larger_size}"
        assert not data1[['sample_name', 'task']].duplicated().any()
        assert not data2[['sample_name', 'task']].duplicated().any()

        combined = data1.merge(data2, on=['sample_name', 'task'], how='inner')
        assert combined.shape[0] == smaller_size, f"Combined: {combined.shape[0]} vs. smaller input size {smaller_size}"

        for c in [c for c in combined.columns if c.startswith('dem_')] + ['targets']:
            c_base = c.replace('_text', '').replace('_audio', '')
            if not all([c in combined.columns for c in [f"{c_base}_audio", f"{c_base}_text"]]):
                continue
            diff = np.abs(combined[f"{c_base}_audio"] - combined[f"{c_base}_text"])
            assert diff.mean() < 0.2

        return combined

    def _col_is_relevant(self, col):
        if col in ['sample_name', 'targets', 'targets_audio', 'targets_text', 'predictions', 'predictions_audio',
                   'predictions_text', 'split_idx', 'split_idx_audio', 'split_idx_text', 'task']:
            return False
        # remove demographic text/audio duplicates
        elif col.startswith('dem_') and col.endswith('_audio'):
            return False
        return True

    def _get_columns(self, combined):
        columns = pd.DataFrame({
            'col': combined.columns,
            'is_relevant': pd.Series(combined.columns).apply(lambda x: self._col_is_relevant(x))
        })

        def get_group(row):
            if row['col'].startswith('node_') and row['col'].endswith('_text'):
                return 'node_text'
            elif row['col'].startswith('node_') and row['col'].endswith('_audio'):
                return 'node_audio'
            elif row['col'].startswith('node_'):
                return 'node'
            elif row['col'].startswith('dem_'):
                return 'demographic'
            elif not row['is_relevant']:
                return None
            elif row['col'].split('_')[0] in ['smile', 'pause', 'phon']:
                return 'acoustic'
            elif row['col'].split('_')[0] in ['lit', 'sung']:
                return 'linguistic'
            elif row['col'].startswith('split_idx'):
                return None
            elif row['col'].startswith('task'):
                return 'task'
            else:
                return 'other'

        columns['group'] = columns.apply(lambda row: get_group(row), axis=1)

        return columns

    def load_combined_data(self):
        combined_data = {}
        for split in range(5):
            train_data_text, test_data_text, val_data_text = self._load_data(self.results_dir_text, split, postfix="text")
            train_data_audio, test_data_audio, val_data_audio = self._load_data(self.results_dir_audio, split, postfix="audio")

            test_data = self._combine_datasets(test_data_text, test_data_audio)
            train_data = self._combine_datasets(train_data_text, train_data_audio)
            val_data = self._combine_datasets(val_data_text, val_data_audio)

            assert np.all(np.abs(test_data['targets_text'] - test_data['targets_audio']) < 1e-5)

            columns = self._get_columns(test_data)
            print("unique groups", columns['group'].dropna().unique())

            train = DataSplit(X=train_data[[c for c in columns.query("is_relevant")['col']]], y=train_data['targets_text'], sample_names=train_data['sample_name'], tasks=train_data['task'])
            test = DataSplit(X=test_data[[c for c in columns.query("is_relevant")['col']]], y=test_data['targets_text'], sample_names=test_data['sample_name'], tasks=test_data['task'])
            val = DataSplit(X=val_data[[c for c in columns.query("is_relevant")['col']]], y=val_data['targets_text'], sample_names=val_data['sample_name'], tasks=val_data['task'])

            combined_data[split] = Data(train=train, val=val, test=test)

        self.columns_lookup = {row['col']: row['group'] for _, row in columns.iterrows() if row['is_relevant']}
        self.unique_groups = columns['group'].dropna().unique()
        self.combined_data = combined_data

        #with open(os.path.join(self.run_parameters.results_dir, 'combined_data.pkl'), 'wb') as f:
        #    pickle.dump(self.combined_data, f)

        combined_data_full = {}
        if self.has_full_data:
            for split in range(1):
                train_data_text, test_data_text, val_data_text = self._load_data(self.full_results_dir_text, split, postfix="text")
                train_data_audio, test_data_audio, val_data_audio = self._load_data(self.full_results_dir_audio, split, postfix="audio")

                test_data = self._combine_datasets(test_data_text, test_data_audio)
                train_data = self._combine_datasets(train_data_text, train_data_audio)
                val_data = self._combine_datasets(val_data_text, val_data_audio)

                columns = self._get_columns(test_data)

                assert np.all(test_data['targets_text'] - test_data['targets_audio'] < 1e-5)

                train = DataSplit(X=train_data[[c for c in columns.query("is_relevant")['col']]], y=train_data['targets_text'], sample_names=train_data['sample_name'], tasks=train_data['task'])
                test = DataSplit(X=test_data[[c for c in columns.query("is_relevant")['col']]], y=test_data['targets_text'], sample_names=test_data['sample_name'], tasks=test_data['task'])
                val = DataSplit(X=val_data[[c for c in columns.query("is_relevant")['col']]], y=val_data['targets_text'], sample_names=val_data['sample_name'], tasks=val_data['task'])

                combined_data_full[split] = Data(train=train, val=val, test=test)

        self.combined_data_full = combined_data_full

        #with open(os.path.join(self.run_parameters.results_dir, 'combined_data_full.pkl'), 'wb') as f:
        #    pickle.dump(self.combined_data_full, f)


    def _group_wise_alpha(self, feature_name, chosen_alphas):
        group = self.columns_lookup.get(feature_name, None)
        if group is None:
            raise ValueError(f"Feature {feature_name} not found in columns_lookup")
        return chosen_alphas[group]

    def _get_best_model(self, X_train, y_train, X_val, y_val, alpha_ranges_logspace, n_runs, split_idx, feature_sets_string):
        def sample_alphas():
            if self.version == 'v8_global_alpha':
                global_low, global_high = np.min([lower for (lower, higher) in alpha_ranges_logspace.values()]), np.max([higher for (lower, higher) in alpha_ranges_logspace.values()])
                global_alpha = np.power(10, np.random.uniform(global_low, global_high))
                return {
                    feature_group: global_alpha for feature_group in alpha_ranges_logspace.keys()
                }
            elif self.version == 'v9_fixed_alpha':
                # based on full sample size analysis with full feature set (kw48/fixed_alpha.ipynb)
                if feature_sets_string == 'demographic':
                    return {'demographic': 69.15143313589714}
                elif feature_sets_string == 'demographic_linguistic_acoustic':
                    return {'demographic': 128.05357011956298, 'acoustic': 3328.8692455737882,
                            'linguistic': 590.1762878476289}
                elif feature_sets_string == 'full':
                    return {'demographic': 96.95765015575152, 'acoustic': 13671.712390328796,
                            'node_audio': 2952.3268666314016, 'node_text': 5961.701631859017,
                            'linguistic': 12968.299877678355}
                elif feature_sets_string == 'demographic_node_text_node_audio':
                    return {'demographic': 112.43784877683561, 'node_audio': 300.1415116761255,
                            'node_text': 1698.702241053147}
                elif feature_sets_string == 'demographic_acoustic_node_audio':
                    return {'demographic': 142.6020006694685, 'acoustic': 9156.994122744134,
                            'node_audio': 355.4854593805735}
                elif feature_sets_string == 'demographic_linguistic_node_text':
                    return {'demographic': 38.81656147793972, 'node_text': 2231.14774975117,
                            'linguistic': 2806.9814028530127}
                else:
                    raise ValueError(f"Unknown feature set: {feature_sets_string}")
            elif self.version == 'v6':
                return {
                    feature_group: np.power(10, np.random.uniform(lower, higher)) for feature_group, (lower, higher) in
                    alpha_ranges_logspace.items()
                }
            else:
                raise ValueError("Invalid version")


        best_r2 = -np.inf
        best_alphas = None
        results = []
        for i in range(n_runs):
            sampled_alphas = sample_alphas()
            model = GeneralizedRidge(alphas=[self._group_wise_alpha(c, sampled_alphas) for c in X_train.columns],
                                     solver="auto")
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)

            r2 = r2_score(y_val, y_pred_val)

            results.append((r2, sampled_alphas))

            if r2 > best_r2:
                best_r2 = r2
                best_alphas = sampled_alphas
                print(f"New best {r2:.5f} ({best_alphas}) (iteration {i}, split {split_idx})")

            if n_runs > 10 and i % (n_runs // 10) == 0 and i > 0:
                print(f"Iteration {i}/{n_runs}, current best R2: {best_r2:.5f} (split {split_idx})")

        # best model
        model = GeneralizedRidge(alphas=[self._group_wise_alpha(c, best_alphas) for c in X_train.columns], solver="auto")
        model.fit(X_train, y_train)

        return model, best_alphas, results

    def find_best_alphas(self, alpha_ranges_logspace, n_runs=None, feature_sets=None):
        if n_runs is None:
            n_runs = self.n_runs
        if self.debug:
            n_runs = 50
        if feature_sets is not None:
            feature_sets_string = "_".join(feature_sets)
        else:
            feature_sets_string = "full"
        print("\n\n--------------------------------")
        print(f"Finding best alphas for generalized ridge regression: feature_set {feature_sets_string}, n_runs {n_runs}, alpha_ranges_logspace {alpha_ranges_logspace}")

        results_dir_here = os.path.join(self.run_parameters.results_dir, feature_sets_string)
        os.makedirs(results_dir_here, exist_ok=True)
        with open(os.path.join(results_dir_here, 'settings.json'), 'w') as f:
            f.write(python_to_json({'n_runs': n_runs, 'feature_sets': feature_sets, 'alpha_ranges_logspace': alpha_ranges_logspace}))

        X_train_0 = self.combined_data[0].train.X
        combined_data_here = {}
        combined_data_full_here = {}
        if feature_sets is not None:
            assert all([f in self.unique_groups for f in feature_sets]), f"Feature set no in valid groups: {[f for f in feature_sets if f not in self.unique_groups]}, should be in {self.unique_groups}"
            features = [c for c in X_train_0.columns if self.columns_lookup.get(c, None) in feature_sets]
            with open(os.path.join(results_dir_here, 'selected_features.json'), 'w') as f:
                f.write(python_to_json({'features': features}))
            print(f"Chose {len(features)} features out of {len(X_train_0.columns)} features")
            for split in range(5):
                combined_data_here[split] = self.combined_data[split].select_features(features)
            if self.has_full_data:
                combined_data_full_here[0] = self.combined_data_full[0].select_features(features)
        else:
            combined_data_here = self.combined_data
            combined_data_full_here = self.combined_data_full

        results_collected = []
        r2_test_collected = []
        r2_val_collected = []
        prediction_test_collected = []
        true_test_collected = []
        prediction_val_collected = []
        true_val_collected = []
        chosen_alphas = {}
        run_times = []
        target_predictions = []
        for split in range(5):
            start_time = time.time()
            print("split {}".format(split))
            X_train, y_train = combined_data_here[split].train.X, combined_data_here[split].train.y
            X_val, y_val = combined_data_here[split].val.X, combined_data_here[split].val.y
            X_test, y_test = combined_data_here[split].test.X, combined_data_here[split].test.y
            print(f"Number of samples: Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            model, best_alphas, results = self._get_best_model(X_train, y_train, X_val, y_val, alpha_ranges_logspace, n_runs, split, feature_sets_string)
            chosen_alphas[split] = best_alphas

            print(f"Best alphas for split {split}: {best_alphas}")
            y_pred_val = model.predict(X_val)
            prediction_val_collected.append(y_pred_val)
            true_val_collected.append(y_val)
            r2_val = r2_score(y_val, y_pred_val)
            r2_val_collected.append(r2_val)

            y_pred_test = model.predict(X_test)
            prediction_test_collected.append(y_pred_test)
            true_test_collected.append(y_test)
            r2_test = r2_score(y_test, y_pred_test)
            r2_test_collected.append(r2_test)
            print(f"... gives val r2={r2_val:.3f}, test r2={r2_test:.3f}")

            for r2, alpha_dict in results:
                results_collected.append({
                    "split": split,
                    "r2": r2,
                    **alpha_dict,
                })

            target_predictions.append(pd.DataFrame({
                'target': y_test,
                'prediction': y_pred_test,
                'sample_name': combined_data_here[split].test.sample_names,
                'task': combined_data_here[split].test.tasks,
                'split': split
            }))

            runtime = time.time() - start_time
            print(f"Runtime for split {split}: {int(runtime // 60)}min {int(runtime % 60)}s")
            run_times.append(runtime)

        target_predictions = pd.concat(target_predictions)
        target_predictions.to_csv(os.path.join(results_dir_here, "target_predictions.csv"), index=False)

        true_val_collected = np.concatenate(true_val_collected)
        prediction_val_collected = np.concatenate(prediction_val_collected)
        true_test_collected = np.concatenate(true_test_collected)
        prediction_test_collected = np.concatenate(prediction_test_collected)

        r2_over_splits_val = np.mean(r2_val_collected)
        r2_over_splits_test = np.mean(r2_test_collected)
        print(f"Avg val r2 over splits (separate alphas per split): {r2_over_splits_val:.3f}")
        print(f"Avg test r2 over splits (separate alphas per split): {r2_over_splits_test:.3f}")

        r2_combined_val = r2_score(true_val_collected, prediction_val_collected)
        r2_combined_test = r2_score(true_test_collected, prediction_test_collected)
        ci_val = bootstrap((true_val_collected, prediction_val_collected), r2_score, paired=True,
                           n_resamples=1000).confidence_interval
        ci_test = bootstrap((true_test_collected, prediction_test_collected), r2_score, paired=True,
                            n_resamples=1000).confidence_interval
        print(f"R2 over combined val splits (separate alphas per split):")
        print(f"- Val: {r2_combined_val:.3f} ({ci_val.low:.3f} - {ci_val.high:.3f})")
        print(f"- Test: {r2_combined_test:.3f} ({ci_test.low:.3f} - {ci_test.high:.3f})")

        df = pd.DataFrame(results_collected)
        df_log = df.copy()
        for col in df.columns:
            if col not in ["r2", 'split']:
                df_log[col] = np.log10(df[col])

        # plot results
        fig = go.Figure()

        for split in range(5):
            for col in df_log.columns:
                if col in ["r2", "split"]:
                    continue
                fig.add_trace(go.Scatter(
                    x=df_log.query("split == @split")[col],
                    y=df_log.query("split == @split")["r2"],
                    mode="markers",
                    name=col,
                    opacity=0.7
                ))

        fig.update_layout(
            title="Validation R² vs log₁₀(α) per Feature Group",
            xaxis_title="log₁₀(α)",
            yaxis_title="R² Score",
            width=900,
            height=500,
            legend_title="Feature Group"
        )

        fig.write_html(os.path.join(results_dir_here, "r2_vs_log_alpha_scatter.html"))

        def smooth_curve(log_alpha, r2_vals, bins=15, aggregator='mean'):
            df = pd.DataFrame({'log_alpha': log_alpha, 'r2': r2_vals})
            df['bin'] = pd.cut(df['log_alpha'], bins=bins)
            if aggregator == 'mean':
                grouped = df.groupby('bin', observed=True)[['log_alpha', 'r2']].mean().reset_index()
            elif aggregator == 'max':
                grouped = df.groupby('bin', observed=True)[['log_alpha', 'r2']].max().reset_index()
            return grouped

        # --- Assign consistent colors for feature groups ---
        color_map = {
            col: color for col, color in zip(
                [c for c in df_log.columns if c != "r2"],
                ['red', 'blue', 'green', 'purple', 'orange', 'teal', 'gold']
            )
        }

        fig = go.Figure()

        # --- Plot MAX ---
        for col in [c for c in df_log.columns if c not in ["r2", "split"]]:
            for split in range(5):
                smooth = smooth_curve(df_log.query("split == @split")[col], df_log.query("split == @split")["r2"],
                                      aggregator='max')
                fig.add_trace(go.Scatter(
                    x=smooth["log_alpha"],
                    y=smooth["r2"],
                    mode="lines+markers",
                    name=f"{col} split{split} (max)",
                    line=dict(color=color_map[col], dash='solid'),
                    marker=dict(symbol='star', color=color_map[col]),
                    opacity=0.8
                ))

        # --- Layout ---
        fig.update_layout(
            title="Smoothed Validation R² vs log₁₀(α) per Feature Group",
            xaxis_title="log₁₀(α)",
            yaxis_title="R² (Mean & Max per Bin)",
            width=900,
            height=550,
            legend_title="Feature Group / Aggregator"
        )

        fig.write_html(os.path.join(results_dir_here, "r2_vs_log_alpha_max.html"))

        # calculate for full data (train / test split)
        if self.has_full_data:
            print("Calculating R2 for full dataset (train / test split)...")
            start_time = time.time()
            X_train, y_train = combined_data_full_here[0].train.X, combined_data_full_here[0].train.y
            X_test, y_test = combined_data_full_here[0].test.X, combined_data_full_here[0].test.y
            X_val, y_val = combined_data_full_here[0].val.X, combined_data_full_here[0].val.y
            model, best_alphas, results = self._get_best_model(X_train, y_train, X_val, y_val, alpha_ranges_logspace, n_runs,"Train/Test", feature_sets_string)
            chosen_alphas["Train/Test"] = best_alphas
            #model = GeneralizedRidge(alphas=[self._group_wise_alpha(c, best_alphas) for c in X_train.columns], solver="auto")
            #model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            r2_full_test = r2_score(y_test, y_pred_test)
            ci_full_test = bootstrap((y_test, y_pred_test), r2_score, paired=True, n_resamples=1000).confidence_interval
            y_pred_val = model.predict(X_val)
            r2_full_val = r2_score(y_val, y_pred_val)
            ci_full_val = bootstrap((y_val, y_pred_val), r2_score, paired=True, n_resamples=1000).confidence_interval
            print(f"R2 full dataset (separate alphas per split):")
            print(f"- Val: {r2_full_val:.3f} ({ci_full_val.low:.3f} - {ci_full_val.high:.3f})")
            print(f"- Test: {r2_full_test:.3f} ({ci_full_test.low:.3f} - {ci_full_test.high:.3f})")
            run_times.append(time.time() - start_time)
            pd.DataFrame({
                'target': y_test,
                'prediction': y_pred_test,
                'sample_name': combined_data_full_here[0].test.sample_names,
                'task': combined_data_full_here[0].test.tasks,
                'split': "Train/Test"
            }).to_csv(os.path.join(results_dir_here, "target_predictions_full.csv"), index=False)
            res_full = {
                'r2_full_test': r2_full_test,
                'r2_full_ci_test': ci_full_test,
                'r2_full_val': r2_full_val,
                'r2_full_ci_val': ci_full_val,
            }

        else:
            print("No full dataset available, skipping full data R2 calculation.")
            res_full = {}

        res = {
            'chosen_alphas': chosen_alphas,
            'r2_over_splits_val': r2_over_splits_val,
            'r2_val_collected': r2_val_collected,
            'r2_over_splits_test': r2_over_splits_test,
            'r2_test_collected': r2_test_collected,
            'r2_combined_val': r2_combined_val,
            'r2_combined_ci_val': f"{ci_val.low:.3f} - {ci_val.high:.3f}",
            'r2_combined_test': r2_combined_test,
            'r2_combined_ci_test': f"{ci_test.low:.3f} - {ci_test.high:.3f}",
            **res_full,
            'feature_sets': feature_sets,
            'n_runs': n_runs,
            'datetime': str(datetime.now()),
            'run_times': run_times,
            # 'test_runs': results_collected.,
        }

        with open(os.path.join(results_dir_here, "results.json"), "w") as f:
            json.dump(res, f, indent=4)
        with open(os.path.join(results_dir_here, "chosen_alphas.json"), "w") as f:
            json.dump(chosen_alphas, f, indent=4)

        self.combine_results()

        return best_alphas, results, chosen_alphas


    def _plot_alphas(self, version):
        combined_results = pd.read_csv(os.path.join(self.run_parameters.results_dir, "combined_results_overview.csv"))
        feature_sets = ['demographic', 'acoustic', 'linguistic', 'node_audio', 'node_text']
        alphas = combined_results.set_index("feature_set").loc[version]
        available_feature_sets = [fs for fs in feature_sets if fs in alphas.index]
        alphas = alphas[available_feature_sets].rename(
            {'node_audio': 'fine-tuning (audio)', 'node_text': 'fine-tuning (text)',
             'linguistic': 'hand-crafted (text)', 'acoustic': 'hand-crafted (audio)'}).dropna()

        # explode values in alphas into dataframe for boxplot
        alphas_expanded = []
        for feature_set, alpha_values in alphas.items():
            alpha_list = [float(a) for a in alpha_values.strip('[]').split(',') if a]
            for alpha in alpha_list:
                alphas_expanded.append({'feature_set': feature_set, 'alpha': alpha})
        alphas_expanded = pd.DataFrame(alphas_expanded).set_index('feature_set')['alpha']

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        cats = list(alphas_expanded.index.unique())
        x_numeric = np.array([cats.index(v) for v in alphas_expanded.index])
        sns.boxplot(x=x_numeric, y=alphas_expanded.values, width=0.3, ax=ax, fliersize=0)
        sns.scatterplot(x=x_numeric + 0.25, y=alphas_expanded.values, color='black', alpha=1, s=60, ax=ax)
        plt.ylabel("Ridge penalty alpha value")
        plt.xlabel("")
        plt.yscale("log")
        plt.ylim(min(1e1, alphas_expanded.min()), max(1e7, alphas_expanded.max()))
        plt.xticks(ticks=range(len(cats)), labels=cats, rotation=45, ha='right')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs(os.path.join(self.run_parameters.results_dir, "chosen_alphas"), exist_ok=True)
        plt.savefig(os.path.join(self.run_parameters.results_dir, "chosen_alphas", f"{version}.png"))
        if version == 'full':
            plt.savefig(os.path.join(self.run_parameters.results_dir, "chosen_alphas_full.png"))
        plt.show()

    def _plot_results(self):
        pass

    def combine_results(self):
        # now reload all data for the overview table

        # iterate through folder
        results = []
        subdirs = [item for item in os.listdir(self.run_parameters.results_dir) if os.path.isdir(os.path.join(self.run_parameters.results_dir, item))]
        subdirs = [s for s in subdirs if s not in ['chosen_alphas']]
        print("Subdirs in results dir:", subdirs)
        for feature_set_string in subdirs:
            results_path = os.path.join(self.run_parameters.results_dir, feature_set_string, "results.json")
            if not os.path.isfile(results_path):
                continue
            with open(results_path, "r") as f:
                jsonload = json.load(f)
            alphas_dict = pd.DataFrame(jsonload['chosen_alphas'].values()).to_dict(orient="list")
            alphas_dict_print = {col: [int(val) for val in alphas_dict[col]] for col in alphas_dict.keys()}
            res_full = {
                "r2 full (val)": f"{jsonload['r2_full_val']:.3f} ({jsonload['r2_full_ci_val'][0]:.3f} - {jsonload['r2_full_ci_val'][1]:.3f})",
                "r2 full (test)": f"{jsonload['r2_full_test']:.3f} ({jsonload['r2_full_ci_test'][0]:.3f} - {jsonload['r2_full_ci_test'][1]:.3f})",
            } if 'r2_full_test' in jsonload else {}
            res = {
                'target': self.target_variable,
                "feature_set": feature_set_string,
                "n_runs": f"{jsonload['n_runs']}",
                "r2 development (val)": f"{jsonload['r2_combined_val']:.3f} ({jsonload['r2_combined_ci_val']})",
                "r2 development (test)": f"{jsonload['r2_combined_test']:.3f} ({jsonload['r2_combined_ci_test']})",
                **res_full,
                'filedate': datetime.fromtimestamp(os.path.getctime(results_path)).strftime('%Y-%m-%d %H:%M:%S'),
                **alphas_dict_print,
            }
            results.append(res)

        pd.DataFrame(results).to_csv(os.path.join(self.run_parameters.results_dir, "combined_results_overview.csv"), index=False)

        for feature_set_string in subdirs:
            self._plot_alphas(feature_set_string)


def run(run_parameters: RunParameters, config: Config, constants: Constants):
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

    model = CombineWavlmDeberta(run_parameters, config, constants)
    model.load_combined_data()

    # baseline: only demographic
    alpha_ranges_logspace = {
        'demographic': (1, 3),
        'task': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['demographic'])

    # baseline: only hand-crafted features
    alpha_ranges_logspace = {
        'demographic': (1, 7),
        'linguistic': (1, 7),
        'acoustic': (1, 7),
        'task': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['demographic', 'linguistic', 'acoustic', 'task'])

    # full model (all feature sets)
    alpha_ranges_logspace = {
        'node_text': (1, 7),
        'node_audio': (1, 7),
        'linguistic': (1, 7),
        'acoustic': (1, 7),
        'demographic': (1, 3),
        'task': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace)

    # baseline: only fine-tuned features
    alpha_ranges_logspace = {
        'demographic': (1, 3),
        'node_text': (1, 7),
        'node_audio': (1, 7),
        'task': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['demographic', 'node_text', 'node_audio', 'task'])

    # baseline: only acoustic domain
    alpha_ranges_logspace = {
        'demographic': (1, 3),
        'acoustic': (1, 7),
        'node_audio': (1, 7),
        'task': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['demographic', 'acoustic', 'node_audio', 'task'])

    # baseline: only text domain
    alpha_ranges_logspace = {
        'demographic': (0, 3),
        'linguistic': (1, 7),
        'node_text': (1, 7),
        'task': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['demographic', 'linguistic', 'node_text', 'task'])

    # baseline: only hand-crafted features, without demographic
    alpha_ranges_logspace = {
        'linguistic': (1, 7),
        'acoustic': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['linguistic', 'acoustic'])

    # baseline: only fine-tuned features, without demographic
    alpha_ranges_logspace = {
        'node_text': (1, 7),
        'node_audio': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['node_text', 'node_audio'])

    # baseline: only wavlm, only deberta, with and without demographics
    alpha_ranges_logspace = {
        'node_audio': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['node_audio'])
    alpha_ranges_logspace = {
        'node_text': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['node_text'])
    alpha_ranges_logspace = {
        'demographic': (0, 3),
        'node_audio': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['demographic', 'node_audio'])
    alpha_ranges_logspace = {
        'demographic': (0, 3),
        'node_text': (1, 7),
    }
    model.find_best_alphas(alpha_ranges_logspace, feature_sets=['demographic', 'node_text'])


    model.combine_results()

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

