import numpy as np
import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, bootstrap
from sklearn import metrics
import pickle
import shap
import statsmodels.api as sm
import time

from xgboost import XGBRegressor

from util.generalized_ridge import GeneralizedRidge

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    pass


from data_analysis.data_preprocessing.feature_outlier_removal_imputation import FeatureOutlierRemovalImputation
from data_analysis.data_preprocessing.feature_standardizer import FeatureStandardizer
from data_analysis.data_preprocessing.percentile_transformer import PercentileTransformer
from data_analysis.model.base_model import BaseModel
from data_analysis.model.helpers.bias_analysis import BiasAnalysis
from util.helpers import python_to_json, prepare_demographics, mean_absolute_percentile_error, \
    plot_target_prediction, get_feature_group_by_column_name, store_timing_information, resolve_cv_fold_assignment


class Regression(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__("Regression", *args, **kwargs)

        def spearman_correlation_metric(y_true, y_pred):
            return spearmanr(y_true, y_pred).statistic
        def pearson_correlation_metric(y_true, y_pred):
            return pearsonr(y_true, y_pred).statistic

        # Note about the difference of explained_variance_score and r2_score: These are the same if the mean prediction
        # is the same as the true mean value. Any sensible regression algorithm should have this property. However, if
        # the distribution of train and test splits are different, we can observe such differences, leading to lower r^2
        # scores than explained variance scores. This is especially the case in cross validation (e.g. 10-fold), where
        # the test sets are quite small. When combining the test sets and calculating r^2 on their union, it will
        # approach the explained_variance_score. See /analyses/kw45/r2_vs_explained_variance.ipynb for details
        # In summary, it's more reasonable to consider explained_variance_score in our examples
        self.metrics = [metrics.explained_variance_score, metrics.mean_absolute_error, metrics.mean_squared_error,
                        metrics.r2_score, pearson_correlation_metric, spearman_correlation_metric, mean_absolute_percentile_error]

        self.target_variable = None
        if self.config.config_model.target_variable is not None:
            self.target_variable = self.config.config_model.target_variable
        assert self.target_variable is not None, f"Target variable missing"

        try:
            self.model_name = self.config.config_model.model_name
        except (AttributeError, KeyError):
            self.model_name = "LinearRegression"
        assert self.model_name in ['RandomForest', 'LinearRegression', 'Lasso', 'MeanPrediction', 'TabPFN', 'RandomSampling', 'SVR', 'LinearSVR', 'GeneralizedRidge', 'Ridge', 'XGBoost']

        try:
            self.cv_splits = self.config.config_model.cv_splits
        except (AttributeError, KeyError):
            self.cv_splits = 10

        # data split version: we have multiple pre-defined splits: same setup but with different random seeds (for robustness analysis), plus a version with 20% validation set
        try:
            self.data_split_version = self.config.config_model.data_split_version
        except (AttributeError, KeyError):
            self.data_split_version = None
        assert self.data_split_version is None or self.data_split_version in ["val20", "resampled"], f"Invalid data_split_version {self.data_split_version}"

        # For wavlm / deberta models, we reserve a validation set (10% of the train set of each split) for early stopping - the models are no trained on these
        # To use a similar setup and train on the same samples, we can run this with the flag use_validation_set, which will
        # split the training set accordingly and make it available e.g. for hyperparameter tuning
        try:
            self.use_validation_set = self.config.config_model.use_validation_set
        except (AttributeError, KeyError):
            self.use_validation_set = False

        try:
            self.calculate_shap = self.config.config_model.calculate_shap
        except (AttributeError, KeyError):
            self.calculate_shap = False  # because it's slow

        try:
            self.log_models_and_data = self.config.config_model.log_models_and_data
        except (AttributeError, KeyError):
            self.log_models_and_data = False  # because it needs a lot of storage

        # Ridge global alpha --> genearlized ridge becomes ridge regression with a single alpha for all features
        try:
            self.ridge_global_alpha = self.config.config_model.ridge_global_alpha
        except (AttributeError, KeyError):
            self.ridge_global_alpha = False

        valid_data_preprocessors = ['Outlier Removal and Imputation', 'Feature Standardizer']
        try:
            self.data_preprocessors = self.config.data_preprocessors
        except (AttributeError, KeyError):
            self.data_preprocessors = []
        assert all([p in valid_data_preprocessors for p in self.data_preprocessors]), f"Invalid data preprocessor: {[p for p in self.data_preprocessors if p not in valid_data_preprocessors]}"

        for p in self.data_preprocessors:
            if p == 'Outlier Removal and Imputation':
                self.feature_outlier_removal_imputation = FeatureOutlierRemovalImputation(config=self.config, constants=self.CONSTANTS, run_parameters=self.run_parameters)
            elif p == 'Feature Standardizer':
                self.feature_standardizer = FeatureStandardizer(config=self.config, constants=self.CONSTANTS, run_parameters=self.run_parameters)
            else:
                raise ValueError(f"Invalid data preprocessor {p}")

        print(f"Using target_variable={self.target_variable}, model_name={self.model_name}, cv_splits={self.cv_splits}, "
              f"data_preprocessors: {self.data_preprocessors}, use_validation_set={self.use_validation_set}, "
              f"data_split_version={self.data_split_version}, ridge_global_alpha={self.ridge_global_alpha}")

        if self.model_name == 'RandomForest':
            try:
                self.random_forest_parameters = vars(self.config.config_model.random_forest_parameters)
            except (AttributeError, KeyError):
                self.random_forest_parameters = dict(n_estimators=500)  # , max_features='sqrt', max_depth=5
            print(f"Using random forest parameters: {self.random_forest_parameters}")

        elif self.model_name == 'SVR':
            try:
                self.svr_parameters = vars(self.config.config_model.svr_parameters)
            except:
                self.svr_parameters = dict(kernel='rbf', C=0.5)
            print(f"Using SVR parameters: {self.svr_parameters}")

        elif self.model_name == 'LinearSVR':
            try:
                self.svr_parameters = vars(self.config.config_model.svr_parameters)
            except:
                self.svr_parameters = dict()
            print(f"Using SVR parameters: {self.svr_parameters}")

        elif self.model_name == 'Ridge':
            try:
                self.ridge_parameters = vars(self.config.config_model.ridge_parameters)
            except:
                self.ridge_parameters = dict(alpha=1)
            print(f"Using SVR parameters: {self.ridge_parameters}")

        elif self.model_name == 'XGBoost':
            try:
                self.xgboost_parameters = vars(self.config.config_model.xgboost_parameters)
            except:
                self.xgboost_parameters = {'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0.01, 'n_estimators': 400, 'min_child_weight': 1, 'max_depth': 4, 'learning_rate': 0.01, 'gamma': 1, 'colsample_bytree': 0.7}
            print(f"Using XGBoost parameters: {self.xgboost_parameters}")

    def _log_results(self, message, filename):
        print(message)
        with open(os.path.join(self.run_parameters.results_dir, filename), "a") as f:
            f.write(message)
            f.write("\n")

    def prepare_data(self):
        if self.model_name in ['LinearRegression', 'Lasso', 'Ridge']:
            assert 'Feature Standardizer' in self.config.data_preprocessors, "Linear Regression requires Standardization of features"

    def _write_correlation_matrix(self, features: pd.DataFrame, file_postfix=""):
        n_features = features.shape[1]
        f = plt.figure(figsize=(n_features*.3+4, n_features*.3))
        correlation_matrix = features.corr()
        plt.matshow(correlation_matrix, fignum=f.number, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
        plt.xticks(range(features.select_dtypes(['number']).shape[1]), features.select_dtypes(['number']).columns, fontsize=14,
                   rotation=45, ha='left', rotation_mode='anchor')
        plt.yticks(range(features.select_dtypes(['number']).shape[1]), features.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, f"feature_correlation{file_postfix}.png"))
        plt.close()

        with open(os.path.join(self.run_parameters.results_dir, f"feature_correlation{file_postfix}.txt"), "w") as file:
            file.write(python_to_json(correlation_matrix))

    def _target_variable_statistics(self, target_variable):
        print("Preprocessed target variable statistics:")
        print(pd.Series(target_variable).describe())
        plt.figure(figsize=(3, 3))
        plt.hist(target_variable, bins=30)
        plt.title(f"Target variable distribution\nMean: {np.mean(target_variable):.2f}, Std: {np.std(target_variable):.2f}")
        plt.savefig(os.path.join(self.run_parameters.results_dir, "target_variable.png"))
        plt.close()

    def _plot_residuals(self, target_and_prediction):
        fig, ax = plt.subplots(figsize=(4, 4))
        sm.qqplot(target_and_prediction[self.target_variable] - target_and_prediction['prediction'], line='s', ax=ax)
        plt.title("QQ plot for residuals (target - prediction)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, "residuals_qqplot.png"))
        plt.close()

    def _plot_correlation_features_vs_target(self, target_variable, predictions, features):
        num_features = features.shape[1]
        height = 10 if num_features < 70 else num_features * 0.13
        plt.figure(figsize=(10, height))
        df = features.copy()
        df[self.target_variable] = target_variable
        df['prediction'] = predictions
        correlations = df.corr(method="spearman")[[self.target_variable, 'prediction']].sort_values(by=self.target_variable, key=abs, ascending=False).iloc[2:]
        sns.heatmap(correlations, annot=True, yticklabels=1, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
        plt.title(f"Spearman correlation of features with target ({self.target_variable}) and prediction")
        plt.tight_layout()
        correlations.to_csv(os.path.join(self.run_parameters.results_dir, "features_vs_target_correlation.csv"))
        plt.savefig(os.path.join(self.run_parameters.results_dir, "features_vs_target_correlation.png"))
        plt.close()

    def _effect_size(self, target_variable, features):
        df = features.copy()
        assert len(target_variable) == features.shape[0], f"{len(target_variable)} != {features.shape[0]}"
        df[self.target_variable] = target_variable
        correlations = df.corr(method="spearman")[[self.target_variable]].sort_values(by=self.target_variable, key=abs, ascending=False).iloc[1:]
        return correlations

    def _plot_effet_sizes_box(self, effect_sizes_df, output_file_name, subtitle, show_significance=False):
        # sort by median effect size over splits / boostrap samples
        median_effect_size = effect_sizes_df.median().sort_values(key=abs, ascending=False)
        # print(median_effect_size)
        sorted_features = list(median_effect_size.index)
        effect_sizes_df = effect_sizes_df[sorted_features]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=(2, 3), sharex=True)
        positions = range(1, len(sorted_features) + 1)
        ax1.plot(positions, median_effect_size.abs(), "-.", label="Median Effect Size")
        ax1.set_ylabel("Effect size\n[Spearman correlation]")
        if show_significance:
            def conf_intvl(values):
                # flip if majority is < 0
                values = values * np.sign(values.mean())
                return np.percentile(values, [2.5, 97.5])
            confidence_intervals = effect_sizes_df.apply(conf_intvl)
            is_significant = confidence_intervals.apply(lambda row: row[0] > 0)
            ax1.fill_between(positions, confidence_intervals.loc[0, :], confidence_intervals.loc[1, :], label="95% confidence interval", alpha=0.5)
            ax1.plot([1, effect_sizes_df.shape[1]], [0, 0], 'k:')
            ax1.axvline(np.argmax(~is_significant)+0.5, c='r', alpha=0.5, linestyle="--", label="Threshold of significance")
            median_effect_size_significance = pd.DataFrame({'median_effect_size': median_effect_size, 'is_significant': is_significant})
            median_effect_size_significance.to_csv(os.path.join(self.run_parameters.results_dir, f"{output_file_name}_significance.csv"))
        ax1.legend()

        ax2.plot([1, effect_sizes_df.shape[1]], [0, 0], 'k:')
        ax2.boxplot(effect_sizes_df, showfliers=False)
        ax2.set_xticks(range(1, effect_sizes_df.shape[1] + 1), effect_sizes_df, rotation=45, ha='right', rotation_mode="anchor")
        ax2.set_ylabel("Effect size\n[Spearman correlation]")
        if show_significance:
            ax2.axvline(np.argmax(~is_significant) + 0.5, c='r', alpha=0.5, linestyle="--", label="Significant")

        fig.suptitle(f'Spearman correlation of features with target ({self.target_variable}) and prediction\n{subtitle}')
        plt.tight_layout()
        effect_sizes_df.to_csv(os.path.join(self.run_parameters.results_dir, f"{output_file_name}.csv"))
        plt.savefig(os.path.join(self.run_parameters.results_dir, f"{output_file_name}.png"))
        plt.close()

    def _plot_effect_sizes_box_cv(self, targets_per_split, features_per_split):
        correlations_per_split = [self._effect_size(target, features) for target, features in zip(targets_per_split, features_per_split)]
        correlations = pd.concat(correlations_per_split, axis=1).T
        self._plot_effet_sizes_box(correlations, output_file_name="effect_sizes_cv", show_significance=False,
                                   subtitle=f"Box plots over {self.cv_splits} CV splits")

    def _plot_effect_sizes_box_bootstrapped(self, target_variable, features):
        assert len(target_variable) == features.shape[0], f"{len(target_variable)} != {features.shape[0]}"
        n = len(target_variable)
        n_bootstrap_samples = 1000
        bootstrap_indices = [np.random.choice(np.arange(n), size=n, replace=True) for _ in range(n_bootstrap_samples)]
        correlations_per_bootstrap = [self._effect_size(target_variable.iloc[bootstrap_idx], features.iloc[bootstrap_idx, :]) for bootstrap_idx in bootstrap_indices]
        correlations = pd.concat(correlations_per_bootstrap, axis=1).T
        self._plot_effet_sizes_box(correlations, output_file_name="effect_sizes_bootstrap", show_significance=True,
                                   subtitle=f"Box plots over {n_bootstrap_samples} bootstrap samples")

    def _preprocess_data(self, X_train, y_train, demographics_train, sample_names_train,
                         X_test, y_test, demographics_test, sample_names_test,
                         X_val, y_val, demographics_val, sample_names_val,
                         split_idx):

        start_time = time.time()
        if 'Outlier Removal and Imputation' in self.data_preprocessors:
            # remove and impute feature outliers
            # note that we take care not to have data leakage here, by fitting on training data only
            self.feature_outlier_removal_imputation.fit(X_train, demographics_train, sample_names_train)
            X_train = self.feature_outlier_removal_imputation.transform(X_train, demographics_train, sample_names_train)
            X_test = self.feature_outlier_removal_imputation.transform(X_test, demographics_test, sample_names_test)
            if X_val is not None:
                X_val = self.feature_outlier_removal_imputation.transform(X_val, demographics_val, sample_names_val)
            if self.log_models_and_data:
                with open(os.path.join(self.run_parameters.results_dir, f'outlier_removal_split{split_idx}.pkl'), 'wb') as f:
                    pickle.dump(self.feature_outlier_removal_imputation, f, pickle.HIGHEST_PROTOCOL)
        store_timing_information(start_time, self.run_parameters.results_dir, f"Data preprocessing split - Outlier Removal and Imputation")

        start_time = time.time()
        if 'Feature Standardizer' in self.data_preprocessors:
            # now we standardize the feature values (mean 0, std 1)
            self.feature_standardizer.fit(X_train)
            X_train = self.feature_standardizer.transform(X_train)
            X_test = self.feature_standardizer.transform(X_test)
            if X_val is not None:
                X_val = self.feature_standardizer.transform(X_val)
            if self.log_models_and_data:
                with open(os.path.join(self.run_parameters.results_dir, f'feature_standardizer_split{split_idx}.pkl'), 'wb') as f:
                    pickle.dump(self.feature_standardizer, f, pickle.HIGHEST_PROTOCOL)
        store_timing_information(start_time, self.run_parameters.results_dir, f"Data preprocessing split - Feature Standardizer")

        return X_train, y_train, X_test, y_test, X_val, y_val


    def run(self):
        demographics, _ = prepare_demographics(self.data.demographics)

        regression_df = pd.concat([
            #self.data.cca_component_scores.reset_index(drop=True),
            #self.data.factor_scores.reset_index(drop=True),
            self.data.factor_scores_theory.reset_index(drop=True),
            self.data.cognitive_overall_score.reset_index(drop=True),
            self.data.acs_outcomes_imputed.reset_index(drop=True),
            self.data.language_task_scores.reset_index(drop=True),
            self.data.features.reset_index(drop=True),
            demographics.reset_index(drop=True),  # demographics are necessary for stratified CV
            self.data.mean_composite_cognitive_score.reset_index(drop=True),  # mean composite score, necessary for stratified CV
            self.data.extra_targets,  # extra targets, such as questionnaire data
        ], axis=1)
        regression_df['sample_name'] = np.array(self.data.sample_names)
        regression_df['task'] = np.array(self.data.tasks)

        X_cols = list(self.data.features.columns)
        assert self.target_variable not in X_cols, f"Target variable {self.target_variable} is in X_cols ({X_cols})"

        columns_to_keep = X_cols + demographics.columns.to_list() + [self.target_variable, 'sample_name', 'mean_composite_cognitive_score', 'task']
        columns_to_keep_no_duplicates = list(set(columns_to_keep))
        regression_df = regression_df[columns_to_keep_no_duplicates]

        # Dropping nan rows
        source_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age', 'mean_composite_cognitive_score']
        disallow_nan_cols = [self.target_variable] + source_stratification_cols
        non_nan_filter = ~regression_df[disallow_nan_cols].isna().any(axis=1)
        nan_row_sample_names = regression_df.loc[~non_nan_filter].sample_name
        n_rows_before = regression_df.shape[0]
        if np.sum(~non_nan_filter) > 0:
            print(f"\n\nATTENTION: null values in {np.sum(~non_nan_filter)} rows / {n_rows_before} (sample names {nan_row_sample_names.to_list()})\n{regression_df[disallow_nan_cols].isna().sum()}\n\n")
        regression_df = regression_df.dropna(subset=disallow_nan_cols)
        regression_df = regression_df.reset_index(drop=True)

        assert np.all(regression_df.index == np.arange(regression_df.shape[0]))  # make sure index is reset

        self._write_correlation_matrix(regression_df[X_cols], file_postfix="_before_preprocessing")

        regression_df = resolve_cv_fold_assignment(
            regression_df, self.data, self.CONSTANTS, self.cv_splits,
            data_split_version=self.data_split_version,
        )

        if self.model_name == 'RandomForest':
            model = RandomForestRegressor(**self.random_forest_parameters)
        elif self.model_name == 'LinearRegression':
            model = LinearRegression()
        elif self.model_name == 'Lasso':
            model = Lasso(alpha=0.05)
        elif self.model_name == 'Ridge':
            model = Ridge(**self.ridge_parameters)
        elif self.model_name == 'MeanPrediction':
            model = DummyRegressor()
        elif self.model_name == 'TabPFN':
            model = TabPFNRegressor()
        elif self.model_name == 'SVR':
            model = SVR(**self.svr_parameters)
        elif self.model_name == 'LinearSVR':
            model = LinearSVR(**self.svr_parameters)
        elif self.model_name == 'GeneralizedRidge':
            model = GeneralizedRidge()
        elif self.model_name == 'XGBoost':
            model = XGBRegressor(**self.xgboost_parameters)
        else:
            raise ValueError(f"Invalid model name {self.model_name}")

        scores_test = defaultdict(list)
        scores_dummy = defaultdict(list)
        scores_train = defaultdict(list)
        shap_values_collected = []
        test_data_collected = []

        def split_Xy(Xy):
            return Xy[X_cols], Xy[self.target_variable], Xy[demographics.columns], Xy['sample_name'], Xy['task']

        # collected data over splits
        sample_names_train_collected, sample_names_test_collected, sample_names_val_collected = [], [], []
        tasks_train_collected, tasks_test_collected, tasks_val_collected = [], [], []
        features_collected, targets_collected, predictions_collected, split_idx_collected = [], [], [], []
        coefficients_collected = []
        for split_idx in range(self.cv_splits):
            print(f"\n\nRunning split {split_idx} / {self.cv_splits}")
            Xy_train = regression_df.query(f"test_split != {split_idx}").reset_index(drop=True)
            if self.use_validation_set:
                # split training into train / validation set
                X_train_shape_before = Xy_train.shape[0]
                if self.cv_splits > 1:
                    Xy_val = Xy_train[Xy_train[f"split{split_idx}_val"] == True].reset_index(drop=True)
                    Xy_train = Xy_train[Xy_train[f"split{split_idx}_val"] == False].reset_index(drop=True)
                elif self.cv_splits == 1:
                    Xy_val  = Xy_train[Xy_train['train_val_split'] == 'val'].reset_index(drop=True)
                    Xy_train = Xy_train[Xy_train['train_val_split'] == 'train'].reset_index(drop=True)
                assert Xy_val.shape[0] > 0
                assert Xy_train.shape[0] > 0
                assert Xy_train.shape[0] + Xy_val.shape[0] == X_train_shape_before, f"{Xy_train.shape[0]} + {Xy_val.shape[0]} != {X_train_shape_before}"
            else:
                Xy_val, X_val, y_val, demographics_val, sample_names_val = None, None, None, None, None
            Xy_test = regression_df.query(f"test_split == {split_idx}").reset_index(drop=True)
            assert Xy_test.shape[0] > 0, "Empty test data? Something seems to be wrong with the data splitting, if cv_splits=1, make sure you use all data, not just development set"
            X_train, y_train, demographics_train, sample_names_train, tasks_train = split_Xy(Xy_train)
            X_test, y_test, demographics_test, sample_names_test, tasks_test = split_Xy(Xy_test)

            if self.use_validation_set:
                X_val, y_val, demographics_val, sample_names_val, tasks_val = split_Xy(Xy_val)

            X_train, y_train, X_test, y_test, X_val, y_val = self._preprocess_data(
                X_train, y_train, demographics_train, sample_names_train,
                X_test, y_test, demographics_test, sample_names_test,
                X_val, y_val, demographics_val, sample_names_val,
                split_idx)

            sample_names_train_collected.append(sample_names_train.tolist())
            sample_names_test_collected.append(sample_names_test.tolist())
            if self.use_validation_set:
                sample_names_val_collected.append(sample_names_val.tolist())

            tasks_train_collected.append(tasks_train.tolist())
            tasks_test_collected.append(tasks_test.tolist())
            if self.use_validation_set:
                tasks_val_collected.append(tasks_val.tolist())

            mean_predictor = np.ones(len(y_test)) * np.mean(y_train)

            # for GeneralizedRidge, find optimal set of alpha values given validation set
            if self.model_name == 'GeneralizedRidge':
                start_time = time.time()
                feature_groups = list(set([get_feature_group_by_column_name(column_name, group_version='global' if self.ridge_global_alpha else None) for column_name in X_train.columns]))
                def sample_alphas():
                    return {
                        feature_group: np.power(10, np.random.uniform(1, 3)) if feature_group == 'demographic' else np.power(10, np.random.uniform(1, 7)) for feature_group in feature_groups    # alphas between 10^-1 and 10^7
                    }
                def get_alpha(column_name, sampled_alphas):
                    feature_group = get_feature_group_by_column_name(column_name, group_version='global' if self.ridge_global_alpha else None)
                    if feature_group is None:
                        raise ValueError(f"Unexpected column name {column_name}, cannot infer feature group for ridge alpha setting")
                    alpha = sampled_alphas.get(feature_group, None)
                    if alpha is None:
                        raise ValueError(f"Missing alpha value for feature {column_name} / feature group {feature_group} in sampled_alphas {sampled_alphas}")
                    return alpha
                
                n_random_search_iterations = 10 if self.config.debug else 1000
                best_alpha_settings = None
                best_score = -np.inf
                #print("Nan:")
                #print(X_train.isna().sum().sort_values(ascending=False))
                print("Starting random search for GeneralizedRidge alphas over", n_random_search_iterations, f"iterations (split {split_idx})")
                start_time_random_search = time.time()
                for random_search_iteration in range(n_random_search_iterations):
                    sampled_alphas = sample_alphas()
                    model.set_alphas(alphas=[get_alpha(c, sampled_alphas) for c in X_train.columns])
                    model.fit(X_train, y_train)
                    y_val_pred = model.predict(X_val)
                    score = metrics.r2_score(y_val, y_val_pred)
                    if score > best_score:
                        best_score = score
                        best_alpha_settings = sampled_alphas
                    if random_search_iteration % (n_random_search_iterations // 10) == 0:
                        runtime_here = time.time() - start_time_random_search
                        print(f"{random_search_iteration} / {n_random_search_iterations}: current best R^2 = {best_score} (runtime so far: {int(runtime_here//60)}min, {int(runtime_here%60)}s) ")
                print(f"Best alpha settings from random search: {best_alpha_settings} with R^2 = {best_score} (Split {split_idx})")
                model.set_alphas(alphas=[get_alpha(c, best_alpha_settings) for c in X_train.columns])
                store_timing_information(start_time, self.run_parameters.results_dir, f'Random search for GeneralizedRidge alphas')

            print("Fitting main model...", end=" ")
            start_time = time.time()
            model.fit(X_train, y_train)  # todo: check if we train on train and val here
            store_timing_information(start_time, self.run_parameters.results_dir, f'Fitting main model')
            y_pred_train = model.predict(X_train)
            start_time = time.time()
            y_pred_test = model.predict(X_test)
            store_timing_information(start_time, self.run_parameters.results_dir, f'Model prediction (inference)')
            r2_here = metrics.r2_score(y_test, y_pred_test)
            print("Test R^2:", r2_here)
            targets_collected.append(y_test)
            predictions_collected.append(y_pred_test)
            split_idx_collected.append(np.ones(y_test.shape) * split_idx)
            features_collected.append(X_test)

            # additional debugging output, if needed
            if self.log_models_and_data:
                X_train.to_csv(os.path.join(self.run_parameters.results_dir, f"X_train_split{split_idx}.csv"), index=False)
                y_train.to_csv(os.path.join(self.run_parameters.results_dir, f"y_train_split{split_idx}.csv"), index=False)
                X_test.to_csv(os.path.join(self.run_parameters.results_dir, f"X_test_split{split_idx}.csv"), index=False)
                y_test.to_csv(os.path.join(self.run_parameters.results_dir, f"y_test_split{split_idx}.csv"), index=False)
                if self.use_validation_set:
                    X_val.to_csv(os.path.join(self.run_parameters.results_dir, f"X_val_split{split_idx}.csv"), index=False)
                    y_val.to_csv(os.path.join(self.run_parameters.results_dir, f"y_val_split{split_idx}.csv"), index=False)
                with open(os.path.join(self.run_parameters.results_dir, f'model_split{split_idx}.pkl'),'wb') as f:
                    pickle.dump(model, f)

            # Compute metrics
            for metric in self.metrics:
                scores_test[metric.__name__].append(metric(y_test, y_pred_test))
                scores_dummy[metric.__name__].append(metric(y_test, mean_predictor))
                scores_train[metric.__name__].append(metric(y_train, y_pred_train))

            if self.calculate_shap:
                # Create the explainer and calculate SHAP values
                if self.model_name in ['RandomForest', 'XGBoost']:
                    explainer = shap.TreeExplainer(model)
                    exlainer_kwargs = {}
                elif self.model_name in ['LinearRegression', 'Lasso', 'GeneralizedRidge', 'Ridge']:
                    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="correlation_dependent")
                    exlainer_kwargs = {}
                else:
                    explainer = shap.Explainer(model.predict, X_train)
                    exlainer_kwargs = {'max_evals': 150}  # PermutationExplainer needs more than the default for many (all our) features
                print("SHAP explainer used:", explainer)

                shap_start_time = time.time()
                shap_values_df = pd.DataFrame(explainer(X_test, **exlainer_kwargs).values, columns=X_test.columns)
                print("Elapsed time for SHAP value computation", time.time() - shap_start_time, "seconds")
                shap_values_collected.append(shap_values_df)

                shap.summary_plot(shap_values_df.to_numpy(), X_test, show=False)
                plt.savefig(os.path.join(self.run_parameters.results_dir, f"shap_split{split_idx}.png"))
                plt.close()


            test_data_collected.append((X_test, y_test))

            if self.model_name in ['Lasso', 'LinearRegression', 'Ridge']:
                coefficients = pd.DataFrame({'feature_name': model.feature_names_in_, f'coefficient_split{split_idx}': model.coef_})
                coefficients = coefficients.sort_values(by=f'coefficient_split{split_idx}', key=abs, ascending=False).set_index('feature_name')
                coefficients_collected.append(coefficients)

                print("Linear model R^2 (for reference, calculated separately)")
                import statsmodels.api as sm
                X1 = sm.add_constant(X_train)
                result = sm.OLS(y_train, X1).fit()
                print("R^2:", result.rsquared, "R^2 adjusted:", result.rsquared_adj)

        for metric in self.metrics:
            metric_name = metric.__name__
            self._log_results(f"CV {metric_name}: {np.mean(scores_test[metric_name]):.3f}+-{np.std(scores_test[metric_name]):.3f}", 'results.txt')
            self._log_results(f"  (Mean predictor: {np.mean(scores_dummy[metric_name]):.3f}+-{np.std(scores_dummy[metric_name]):.3f})", 'results.txt')
            self._log_results(f"  (Train: {np.mean(scores_train[metric_name]):.3f}+-{np.std(scores_train[metric_name]):.3f})", 'results.txt')

        with open(os.path.join(self.run_parameters.results_dir, 'scores.json'), "w") as f:
            json.dump({'scores_test': scores_test, 'scores_dummy': scores_dummy, 'scores_train': scores_train}, f)
        with open(os.path.join(self.run_parameters.results_dir, 'sample_names.json'), "w") as f:
            sample_names_dict = {'sample_names_train': sample_names_train_collected, 'sample_names_test': sample_names_test_collected}
            if self.use_validation_set:
                sample_names_dict['sample_names_val'] = sample_names_val_collected
            json.dump(sample_names_dict, f)


        features_test = pd.DataFrame(pd.concat(features_collected).reset_index(drop=True))
        target_and_prediction = pd.DataFrame({
            self.target_variable: np.concatenate(targets_collected),
            'prediction': np.concatenate(predictions_collected),
            'split_idx': np.concatenate(split_idx_collected),
            'sample_name': np.concatenate(sample_names_test_collected),
            'task': np.concatenate(tasks_test_collected),
            #'n_words_collected': features_test['lit_n_words'],
            })
        plot_target_prediction(target_and_prediction, target_variable=self.target_variable,
                               fig_store_path=os.path.join(self.run_parameters.results_dir, "prediction_vs_target.png"),
                               csv_store_path=os.path.join(self.run_parameters.results_dir,
                                                           "prediction_and_target.csv"))
        self._plot_residuals(target_and_prediction)
        self._target_variable_statistics(target_and_prediction[self.target_variable])
        self._write_correlation_matrix(features_test, file_postfix="")

        # store data to disk
        regression_data_collected = pd.concat((
            target_and_prediction,
            features_test
        ), axis=1)
        regression_data_collected.to_csv(os.path.join(self.run_parameters.results_dir, "regression_data.csv"), index=False)

        self._plot_correlation_features_vs_target(target_and_prediction[self.target_variable], target_and_prediction['prediction'], features_test)
        self._plot_effect_sizes_box_cv(targets_collected, features_collected)
        self._plot_effect_sizes_box_bootstrapped(target_and_prediction[self.target_variable], features_test)

        # write shap values
        if self.calculate_shap:
            if len(shap_values_collected) > 0:
                shap_values_collected_with_split_idx = []
                for i, shap_values in enumerate(shap_values_collected):
                    shap_values_with_split_idx = shap_values.copy()
                    shap_values_with_split_idx['split_idx'] = i
                    shap_values_collected_with_split_idx.append(shap_values_with_split_idx)
                shap_values_combined = pd.concat(shap_values_collected_with_split_idx)
                test_data_X = pd.concat([X for (X, y) in test_data_collected])
                shap.summary_plot(shap_values_combined.drop(columns=['split_idx']).to_numpy(), test_data_X, show=False)
                plt.savefig(os.path.join(self.run_parameters.results_dir, "shap.png"))
                plt.close()
                with open(os.path.join(self.run_parameters.results_dir, "shap_values.txt"), "w") as file:
                    file.write(python_to_json(shap_values_combined))
                shap_values_combined.to_csv(os.path.join(self.run_parameters.results_dir, "shap_values.csv"), index=False)

        # write coefficients
        if len(coefficients_collected) > 0:
            coefficients_all = pd.concat(coefficients_collected, axis=1)
            coefficients_stats = coefficients_all.apply(lambda row: pd.Series({'mean': row.mean(), 'std': row.std()}), axis=1).sort_values(by='mean', ascending=False, key=abs)
            coefficients_all = coefficients_all.loc[coefficients_stats.index]  # sort the same
            coefficients_all.reset_index().to_csv(os.path.join(self.run_parameters.results_dir, "coefficients.csv"), index=False)
            num_coefficients = len(coefficients)
            width = 10 if num_coefficients < 70 else num_coefficients * 0.15
            plt.figure(figsize=(width, 5))
            plt.axhline(0, xmin=0, xmax=len(coefficients_all.index), linestyle="--", color="k", alpha=0.4)
            plt.boxplot(coefficients_all.T)
            plt.xticks(np.arange(len(coefficients_all.index))+1, coefficients_all.index, rotation=45, ha='right', rotation_mode="anchor")
            plt.title(f"Regression coefficients (over {self.cv_splits} splits) [target={self.target_variable}]")
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_parameters.results_dir, "coefficients.png"))
            plt.close()

        # run bias analysis
        bias_analysis = BiasAnalysis(self.data, self.target_variable)
        for metric_name in ['explained_variance_score']:#, 'spearman_correlation', 'mean_absolute_error', 'mean_absolute_percentile_error']:
            bias_analysis.bias_analysis(np.concatenate(predictions_collected),
                                        np.concatenate(targets_collected),
                                        np.concatenate(sample_names_test_collected),
                                        metric_name=metric_name,
                                        plot_path=os.path.join(self.run_parameters.results_dir, f"bias_{metric_name}.png"))


        # store data
        if self.log_models_and_data:
            self.data.store_to_disk(os.path.join(self.run_parameters.results_dir, "data"))

