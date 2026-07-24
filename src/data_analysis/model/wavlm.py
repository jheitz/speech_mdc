"""WavLM model for fine-tuning on audio data directly.

This model fine-tunes the WavLM model from Hugging Face on audio data directly,
instead of using linguistic and acoustic features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import os
import json
import shutil
from typing import Optional, Tuple, Union
import time
import pickle
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, WavLMConfig
from transformers.models.wavlm.modeling_wavlm import _HIDDEN_STATES_START_POSITION
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset as HuggingFaceDataset
from datasets import Audio as HuggingFaceAudio

from data_analysis.data_preprocessing.feature_outlier_removal_imputation import FeatureOutlierRemovalImputation
from data_analysis.data_preprocessing.feature_standardizer import FeatureStandardizer
from data_analysis.model.base_model import BaseModel
from data_analysis.dataloader.dataset import Dataset
from util.helpers import prepare_demographics, mean_absolute_percentile_error, plot_target_prediction, deep_dict_diff, \
    store_timing_information, resolve_cv_fold_assignment
from config.config import Config


@dataclass
class SequenceClassifierOutputWithFeatures(SequenceClassifierOutput):
    features: Optional[torch.FloatTensor] = None


class WavLMForSequenceClassificationExtended(WavLMForSequenceClassification):
    """WavLMForSequenceClassificationExtended, which exposes the pooled_layer (interpreted as features) """
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithFeatures(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            features=pooled_output,
        )



class WavLMWithFeatures(WavLMForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        # Save external feature dimension --> from config
        try:
            self.external_feature_dim = config.external_feature_dim
        except AttributeError:
            raise AssertionError("external_feature_dim should be specified in config")

        # replace classifier to handle concatenated features
        self.classifier = nn.Linear(config.classifier_proj_size + self.external_feature_dim, config.num_labels)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        external_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # here the external features are concatenated
        external_features = external_features.to(pooled_output.device, dtype=pooled_output.dtype)
        combined = torch.cat([pooled_output, external_features], dim=-1)

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithFeatures(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            features=combined,
        )



class WavLMModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__("WavLM", *args, **kwargs)

        def spearman_correlation_metric(y_true, y_pred):
            return spearmanr(y_true, y_pred).statistic
        def pearson_correlation_metric(y_true, y_pred):
            return pearsonr(y_true, y_pred).statistic

        # Same metrics as in Regression model
        self.metrics = [metrics.explained_variance_score, metrics.mean_absolute_error, metrics.mean_squared_error,
                        metrics.r2_score, pearson_correlation_metric, spearman_correlation_metric, mean_absolute_percentile_error]

        self.target_variable = None
        if self.config.config_model.target_variable is not None:
            self.target_variable = self.config.config_model.target_variable
        assert self.target_variable is not None, f"Target variable missing"

        try:
            self.cv_splits = self.config.config_model.cv_splits
        except (AttributeError, KeyError):
            self.cv_splits = 5

        # data split version: we have multiple pre-defined splits: same setup but with different random seeds (for robustness analysis), plus a version with 20% validation set
        try:
            self.data_split_version = self.config.config_model.data_split_version
        except (AttributeError, KeyError):
            self.data_split_version = None
        assert self.data_split_version is None or self.data_split_version in ["val20", "resampled", 'random'], f"Invalid data_split_version {self.data_split_version}"

        # whether to use demographic residuals as target variable (i.e. regress out demographics first)
        try:
            self.use_demographic_residuals = self.config.config_model.use_demographic_residuals
        except (AttributeError, KeyError):
            self.use_demographic_residuals = False

        # analogous: feature residuals
        try:
            self.use_feature_residuals = self.config.config_model.use_feature_residuals
        except (AttributeError, KeyError):
            self.use_feature_residuals = False

        assert not (self.use_demographic_residuals and self.use_feature_residuals), "Cannot use both demographic and feature residuals"

        try:
            self.only_n_splits = self.config.config_model.only_n_splits
        except (AttributeError, KeyError):
            self.only_n_splits = self.cv_splits

        try:
            self.data_preprocessors = self.config.data_preprocessors
        except (AttributeError, KeyError):
            self.data_preprocessors = []

        for p in self.data_preprocessors:
            if p == 'Outlier Removal and Imputation':
                self.feature_outlier_removal_imputation = FeatureOutlierRemovalImputation(config=self.config, constants=self.CONSTANTS, run_parameters=self.run_parameters)
            elif p == 'Feature Standardizer':
                self.feature_standardizer = FeatureStandardizer(config=self.config, constants=self.CONSTANTS, run_parameters=self.run_parameters)
            elif p == 'Percentile Transformation':
                raise NotImplementedError("Percentile Transformation not implemented for WavLM model")
            else:
                raise ValueError(f"Invalid data preprocessor {p}")


        try:
            self.wavlm_model_name = self.config.config_model.wavlm_model_name
        except (AttributeError, KeyError):
            self.wavlm_model_name = "microsoft/wavlm-base"

        try:
            self.batch_size = self.config.config_model.batch_size
        except (AttributeError, KeyError):
            self.batch_size = 2

        try:
            self.learning_rate = float(self.config.config_model.learning_rate)
        except (AttributeError, KeyError):
            self.learning_rate = 5e-5

        try:
            self.dropout = float(self.config.config_model.dropout)
        except (AttributeError, KeyError):
            self.dropout = 0

        try:
            self.num_train_epochs = self.config.config_model.num_train_epochs
        except (AttributeError, KeyError):
            self.num_train_epochs = 60

        # Limit audio length to avoid memory issues
        try:
            self.max_audio_length_seconds = self.config.config_model.max_audio_length_seconds
        except (AttributeError, KeyError):
            self.max_audio_length_seconds = 30

        # classifier_proj_size --> for the regression head on top of wavlm
        try:
            self.classifier_proj_size = self.config.config_model.classifier_proj_size
        except (AttributeError, KeyError):
            self.classifier_proj_size = 256  # default in WavLMForSequenceClassification

        try:
            self.use_lora = self.config.config_model.use_lora
        except (AttributeError, KeyError):
            self.use_lora = False

        try:
            self.learning_rate_lora = float(self.config.config_model.learning_rate_lora)
        except (AttributeError, KeyError):
            self.learning_rate_lora = 1e-3

        if self.learning_rate_lora < self.learning_rate:
            raise ValueError(f"learning_rate_lora {self.learning_rate_lora} should be >= learning_rate {self.learning_rate}")

        try:
            # which modules to apply LoRA to:
            # options: attention_light, attention_full
            self.lora_modules = self.config.config_model.lora_modules
        except (AttributeError, KeyError):
            self.lora_modules = 'attention_light'

        try:
            self.lora_r = self.config.config_model.lora_r
        except (AttributeError, KeyError):
            self.lora_r = 16

        try:
            self.lora_dropout = self.config.config_model.lora_dropout
        except (AttributeError, KeyError):
            self.lora_dropout = 0.05

        try:
            self.use_external_features = self.config.config_model.use_external_features
        except (AttributeError, KeyError):
            self.use_external_features = False

        try:
            self.store_models = self.config.config_model.store_models
        except (AttributeError, KeyError):
            self.store_models = self.use_lora

        # r2 threshold, used for hyperparameter testing. if the mean r2 over the first two splits is below this
        # threshold, the training is stopped early. this is to speed up hyperparameter testing
        try:
            self.r2_threshold = self.config.config_model.r2_threshold
        except (AttributeError, KeyError):
            self.r2_threshold = None

        try:
            self.pretrained_model = self.config.config_model.pretrained_model
            assert os.path.exists(self.pretrained_model), f"Pretrained model dir does not exist at {self.pretrained_model}"
            assert self.num_train_epochs == 0, f"When using a pretrained_model, num_train_epochs should be 0 (no further training), but is {self.num_train_epochs}"
            # assert the pretrained model config is the same as the current config
            pretrained_config_yaml = [f for f in os.listdir(self.pretrained_model) if f.endswith(".yaml")][0]
            pretrained_config = Config.from_yaml(os.path.join(self.pretrained_model, pretrained_config_yaml)).to_dict()
            current_config = self.config.to_dict()
            del current_config['config_model']['pretrained_model']  # ignore this field for comparison
            del current_config['config_model']['num_train_epochs']  # ignore this field for comparison
            del current_config['config_data']['task']  # ignore this field for comparison
            config_diff = deep_dict_diff(current_config, pretrained_config)
            if len(set(config_diff.keys()) - set(['config_model.pretrained_model', 'config_model.num_train_epochs', 'config_data.task'])) > 0:
                raise ValueError("Current config does not match pretrained model config:\n" + json.dumps(config_diff, indent=4))
        except (AttributeError, KeyError):
            self.pretrained_model = None

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print(f"Using target_variable={self.target_variable}, wavlm_model_name={self.wavlm_model_name}, "
              f"cv_splits={self.cv_splits}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, "
              f"num_train_epochs={self.num_train_epochs}, max_audio_length_seconds={self.max_audio_length_seconds}, "
              f"device={self.device}, dropout={self.dropout}, use_lora={self.use_lora}, "
              f"learning_rate_lora={self.learning_rate_lora}, store_models={self.store_models}, "
              f"classifier_proj_size={self.classifier_proj_size}, use_external_features={self.use_external_features},"
              f"pretrained_model={self.pretrained_model}, ")

        # Initialize feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wavlm_model_name)



    def _log_results(self, message, filename):
        print(message)
        with open(os.path.join(self.run_parameters.results_dir, filename), "a") as f:
            f.write(message)
            f.write("\n")

    def prepare_data(self):
        # No specific preparation needed on the whole dataset, prep is done on each split
        pass

    def _prepare_dataset(self, data_df):
        features_cols = list(self.data.features.columns) if self.external_features else []

        if self.use_demographic_residuals or self.use_feature_residuals:
            target_variable = "target_residuals"
        else:
            target_variable = self.target_variable

        data_df = data_df.copy()[['audio_file', target_variable] + features_cols].rename(columns={target_variable: 'target'})

        dataset = HuggingFaceDataset.from_pandas(data_df)
        dataset = dataset.cast_column("audio_file", HuggingFaceAudio(sampling_rate=16000, mono=True))

        def preprocess(batch):
            audio_array = batch["audio_file"]["array"]
            features = self.feature_extractor(
                audio_array,
                sampling_rate=16000,
                padding="max_length",
                truncation=True,
                max_length=16000 * self.max_audio_length_seconds,
                return_attention_mask=True,
                return_tensors="pt",
            )
            batch["input_values"] = features["input_values"][0]
            batch["attention_mask"] = features["attention_mask"][0]
            if self.external_features:
                batch["external_features"] = torch.tensor([batch[c] for c in self.data.features.columns], dtype=torch.float)
            return batch

        print("Preprocessing audio files")
        dataset = dataset.map(preprocess, remove_columns=["audio_file"], num_proc=1).with_format("torch")  #, num_proc=os.cpu_count()

        return dataset

    def _create_model_and_optimizer(self, split_idx):
        """Create WavLM model for regression, with LoRA fine-tuning."""

        if self.lora_modules == 'attention_light':
            target_modules = ["q_proj", "v_proj"]
        elif self.lora_modules == 'attention_full':
            target_modules = ["q_proj","k_proj","v_proj","out_proj"]
        else:
            raise ValueError(f"Unknown lora_modules {self.lora_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,  # alpha constant, according to https://github.com/spacepxl/demystifying-sd-finetuning?tab=readme-ov-file, changing alpha is similar to changing learning rate
            lora_dropout=self.lora_dropout, # dropout to apply in the LoRA layers
            r=self.lora_r, # rank controls the capacity of the adaptation, hyperparameter to tune
            target_modules=target_modules,
            bias="none", inference_mode=False, use_rslora=True,
            modules_to_save=["projector", "classifier"],  # keep regression head trainable
        )

        config = WavLMConfig.from_pretrained(self.wavlm_model_name, num_labels=1, problem_type="regression",
                                             hidden_dropout=self.dropout, activation_dropout=self.dropout,
                                             classifier_proj_size=self.classifier_proj_size)

        if self.external_features and self.use_external_features:
            config.external_feature_dim = self.data.features.shape[1]
            WavLMRegressionClass = WavLMWithFeatures
        else:
            WavLMRegressionClass = WavLMForSequenceClassificationExtended

        wavlm_regression = WavLMRegressionClass.from_pretrained(self.wavlm_model_name, config=config)

        if self.use_lora:
            peft_model = get_peft_model(wavlm_regression, lora_config)

            # get the base model from the PeftModelForFeatureExtraction model
            # the PEFT model is a wrapper but assumes certain kwargs that work only for text (input_ids, not wavlm with input_values), that's why we need to get the base model
            # the basemodel does contain the LoRA injected parameters
            model = peft_model.base_model

            # LoRA actually injected?
            injected = [n for n, _ in model.named_parameters() if "lora_" in n]
            print("LoRA params:", len(injected));
            assert injected, "No LoRA params — target_modules mismatch."

            # function to store trained parameters (just relevant ones, not entire wavlm model)
            model_save_fct = lambda save_dir: peft_model.save_pretrained(save_dir, safe_serialization=True)

            # model to restore from stored parameters
            def model_restore_fct(restore_dir):
                wavlm_regression = WavLMRegressionClass.from_pretrained(self.wavlm_model_name, config=config)
                peft_model = PeftModel.from_pretrained(wavlm_regression, restore_dir)
                model = peft_model.base_model
                model.to(self.device)
                return model

        else:
            model = wavlm_regression

            # function to store trained parameters (just relevant ones, not entire wavlm model)
            model_save_fct = lambda save_dir: model.save_pretrained(save_dir, safe_serialization=True)

            # model to restore from stored parameters
            model_restore_fct = lambda restore_dir: WavLMRegressionClass.from_pretrained(restore_dir).to(self.device)

        try:
            print("Trainable parameters:")
            model.print_trainable_parameters()
        except Exception:
            print("Could not print trainable parameters")
            pass

        if self.pretrained_model:
            print(f"Loading pretrained model from {self.pretrained_model}...")
            model = model_restore_fct(os.path.join(self.pretrained_model, f"best_model_split{split_idx}"))

        print("Model to fine-tune:")
        print(model)
        print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        model = model.to(self.device)

        if self.use_lora:
            lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
            head_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and ("classifier.modules_to_save" in n
                                                   or "projector.modules_to_save" in n)]

            optimizer = torch.optim.AdamW(
                [
                    {"params": lora_params, "lr": self.learning_rate_lora},
                    {"params": head_params, "lr": self.learning_rate},
                ],
                weight_decay=0.01,
            )
            print("len(lora_params)", len(lora_params), "len(head_params)", len(head_params))
            print("len([p for p in model.parameters() if p.requires_grad])", len([p for p in model.parameters() if p.requires_grad]))
        else:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=0.01)


        return model, optimizer, model_save_fct, model_restore_fct

    def _get_dataset_batches(self, dataset, shuffle=True):
        if shuffle:
            indices = np.random.permutation(len(dataset))
        else:
            indices = np.arange(len(dataset))
        batch_indices_list = [indices[start:end] for start, end in zip(
            range(0, len(indices), self.batch_size),
            range(self.batch_size, len(indices) + self.batch_size, self.batch_size)
        )]

        for batch_indices in batch_indices_list:
            yield dataset[batch_indices]

    def _train_and_evaluate(self, train_dataset, val_dataset, test_dataset, split_idx):
        """Train and evaluate WavLM model."""
        # Create model
        model, optimizer, model_save_fct, model_restore_fct = self._create_model_and_optimizer(split_idx)

        loss_fn = nn.MSELoss()

        def predict(model, batch):
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device).bool()
            targets = batch["target"].to(self.device).float().view(-1)
            external_features = batch["external_features"].to(self.device) if self.external_features else None
            if self.use_external_features:
                model_output = model(input_values=input_values, attention_mask=attention_mask, external_features=external_features)
                feature_values = model_output.features
            else:
                model_output = model(input_values=input_values, attention_mask=attention_mask)
                if self.external_features:
                    feature_values = torch.cat((model_output.features, external_features), dim=1)
                else:
                    feature_values = model_output.features
            predictions = model_output.logits.view(-1)
            assert predictions.shape == targets.shape, f"Predictions shape {predictions.shape} does not match targets shape {targets.shape}"
            return predictions, targets, feature_values

        def get_predictions(model, dataset, return_feature_values=False):
            model.eval()
            predictions_collected = []
            targets_collected = []
            losses = []
            feature_values_collected = []
            with torch.no_grad():
                for batch in self._get_dataset_batches(dataset, shuffle=False):
                    predictions, targets, feature_values = predict(model, batch)
                    loss = loss_fn(predictions, targets)
                    losses.append(loss.item())
                    predictions_collected.extend(predictions.cpu().tolist())
                    targets_collected.extend(targets.cpu().tolist())
                    feature_values_collected.append(feature_values.cpu().numpy())
            if return_feature_values:
                return predictions_collected, targets_collected, losses, np.concatenate(feature_values_collected, axis=0)
            else:
               return predictions_collected, targets_collected, losses

        start_time = time.time()
        losses_file = os.path.join(self.run_parameters.results_dir, f"losses_split{split_idx}.csv")
        with open(losses_file, "w") as fd:
            fd.write(f"epoch,train_loss,val_loss,test_loss,{','.join([metric.__name__ for metric in self.metrics])}\n")

        epoch_predictions_dir = os.path.join(self.run_parameters.results_dir, f"epoch_predictions_split{split_idx}")
        os.makedirs(epoch_predictions_dir, exist_ok=True)

        lowest_val_loss = ({'epoch': -1, 'loss': np.inf})

        for epoch in range(self.num_train_epochs):
            model.train()
            print(f"Epoch {epoch + 1}/{self.num_train_epochs}")
            losses_epoch = []
            start_time_epoch = time.time()
            for batch in self._get_dataset_batches(train_dataset, shuffle=True):
                predictions, targets, feature_values = predict(model, batch)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses_epoch.append(loss.item())
            store_timing_information(start_time_epoch, self.run_parameters.results_dir, f'Pure epoch learning time')
            elapsed_time = time.time() - start_time
            elapsed_time_h = int(elapsed_time // 3600)
            elapsed_time_min = int((elapsed_time % 3600) / 60)
            elapsed_time_s = int(elapsed_time % 60)
            print(f"  Epoch {epoch + 1} finished in {elapsed_time_h:02d}:{elapsed_time_min:02d}:{elapsed_time_s:02d} ({elapsed_time:.2f}s)")
            print(f"  Finished ({elapsed_time:.2f}s)")
            print(f"  Training loss (mean over {len(train_dataset) // self.batch_size} batches): {np.mean(losses_epoch):.4f}")

            # calculate loss on test set
            model.eval()
            val_predictions, val_targets, val_losses = get_predictions(model, val_dataset)
            print(f"  Validation loss (mean over {len(val_dataset) // self.batch_size} batches): {np.mean(val_losses):.4f}")
            test_predictions, test_targets, test_losses = get_predictions(model, test_dataset)
            test_metrics_epoch = [f"{metric(test_targets, test_predictions):.4f}" for metric in self.metrics]
            print(f"  Test loss (mean over {len(test_dataset) // self.batch_size} batches): {np.mean(test_losses):.4f}")
            with open(losses_file, 'a') as fd:
                fd.write(f"{epoch + 1},{np.mean(losses_epoch):.4f},{np.mean(val_losses):.4f},{np.mean(test_losses):.4f},{','.join(test_metrics_epoch)}\n")

            #pd.DataFrame({'test_predictions': test_predictions, 'test_targets': test_targets}).to_csv(os.path.join(epoch_predictions_dir, f"epoch_{epoch}.csv"), index=False)

            model_path = os.path.join(self.run_parameters.results_dir, f"best_model_split{split_idx}")
            os.makedirs(model_path, exist_ok=True)
            if np.mean(val_losses) < lowest_val_loss['loss']:
                lowest_val_loss = {'epoch': epoch, 'loss': np.mean(val_losses)}
                model_save_fct(model_path)
                print(f"  New lowest validation loss, saving model to {model_path}")
                #pd.DataFrame({'test_predictions': test_predictions, 'test_targets': test_targets}).to_csv(os.path.join(epoch_predictions_dir, f"epoch_{epoch}.csv"), index=False)

            if lowest_val_loss['epoch'] < epoch - 15 or epoch == self.num_train_epochs - 1:
                print(f"(Early) stopping at epoch {epoch} (lowest test loss at epoch {lowest_val_loss['epoch']}, {epoch - lowest_val_loss['epoch']} epochs ago)")
                # load best model
                model = model_restore_fct(model_path)
                print(f"  Loaded model from {model_path} for evaluation")
                test_predictions, test_targets, test_losses = get_predictions(model, test_dataset)
                test_metrics_epoch = [f"{metric(test_targets, test_predictions):.4f}" for metric in self.metrics]
                with open(losses_file, 'a') as fd:
                    fd.write(f"Best,null,null,{np.mean(test_losses):.4f},{','.join(test_metrics_epoch)}\n")

                if not self.store_models:
                    shutil.rmtree(model_path)  # delete model to save space

                break


        def load_and_store_features(model, dataset, path):
            predictions, targets, losses, feature_values = get_predictions(model, dataset, return_feature_values=True)
            if self.external_features:
                feature_values_df = pd.DataFrame(feature_values, columns=[f"node_{i}" for i in range(feature_values.shape[1] - len(self.data.features.columns))] + list(self.data.features.columns))
            else:
                feature_values_df = pd.DataFrame(feature_values, columns=[f"node_{i}" for i in range(feature_values.shape[1])])
            feature_values_df['predictions'] = predictions
            feature_values_df['targets'] = targets
            feature_values_df.to_csv(os.path.join(self.run_parameters.results_dir, path), index=False)
            return predictions, targets

        model.eval()
        start_time_inference = time.time()
        test_predictions, test_targets = load_and_store_features(model, test_dataset, f"test_feature_values_split{split_idx}.csv")
        store_timing_information(start_time_inference, self.run_parameters.results_dir, f'Pure inference time test set')
        load_and_store_features(model, val_dataset, f"val_feature_values_split{split_idx}.csv")
        train_predictions, train_targets = load_and_store_features(model, train_dataset, f"train_feature_values_split{split_idx}.csv")

        return model, test_predictions, test_targets, train_predictions, train_targets

    def _preprocess_features(self, train_df, test_df, split_idx):
        demographics_cols = prepare_demographics(self.data.demographics)[0].columns
        X_train, demographics_train = train_df[self.data.features.columns], train_df[demographics_cols]
        X_test, demographics_test = test_df[self.data.features.columns], test_df[demographics_cols]

        if 'Outlier Removal and Imputation' in self.data_preprocessors:
            # remove and impute feature outliers
            # note that we take care not to have data leakage here, by fitting on training data only
            self.feature_outlier_removal_imputation.fit(X_train, demographics_train, None)
            X_train = self.feature_outlier_removal_imputation.transform(X_train, demographics_train, None)
            X_test = self.feature_outlier_removal_imputation.transform(X_test, demographics_test, None)
            #if self.log_models_and_data:
            #    with open(os.path.join(self.run_parameters.results_dir, f'outlier_removal_split{split_idx}.pkl'), 'wb') as f:
            #        pickle.dump(self.feature_outlier_removal_imputation, f, pickle.HIGHEST_PROTOCOL)

        if 'Feature Standardizer' in self.data_preprocessors:
            # now we standardize the feature values (mean 0, std 1)
            self.feature_standardizer.fit(X_train)
            X_train = self.feature_standardizer.transform(X_train)
            X_test = self.feature_standardizer.transform(X_test)
            #if self.log_models_and_data:
            #    with open(os.path.join(self.run_parameters.results_dir, f'feature_standardizer_split{split_idx}.pkl'), 'wb') as f:
            #        pickle.dump(self.feature_standardizer, f, pickle.HIGHEST_PROTOCOL)

        train_df[list(X_train.columns)] = X_train
        test_df[list(X_train.columns)] = X_test

        # some features might have been deleted due to outlier removal -> set them to 0
        deleted_features = [c for c in self.data.features.columns if c not in X_train.columns]
        train_df[deleted_features] = 0
        test_df[deleted_features] = 0

        return train_df, test_df

    def _prepare_residuals(self, train_df, test_df, split_idx):
        # Define pipeline: standardize then regress
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression())
        ])

        if self.use_demographic_residuals:
            demographics_cols = prepare_demographics(self.data.demographics)[0].columns
            X_train = train_df[demographics_cols]
            X_test = test_df[demographics_cols]
        elif self.use_feature_residuals:
            X_train = train_df[self.data.features.columns]
            X_test = test_df[self.data.features.columns]
        else:
            raise ValueError("Either use_demographic_residuals or use_feature_residuals must be True")

        pipe.fit(X_train, train_df[self.target_variable])
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        train_residuals = train_df[self.target_variable] - train_pred
        test_residuals = test_df[self.target_variable] - test_pred
        train_df['target_residuals'] = train_residuals
        test_df['target_residuals'] = test_residuals

        residuals_model_results_dir = os.path.join(self.run_parameters.results_dir, 'residuals_model')
        os.makedirs(residuals_model_results_dir, exist_ok=True)

        with open(os.path.join(residuals_model_results_dir, f'pipe_split{split_idx}.pkl'), 'wb') as f:
            pickle.dump(pipe, f, pickle.HIGHEST_PROTOCOL)
        linreg_coef_df = pd.DataFrame({'feature': list(X_train.columns) + ['intercept'],
                                       'coefficient': list(pipe.named_steps["linreg"].coef_) + [pipe.named_steps["linreg"].intercept_]})
        linreg_coef_df.to_csv(os.path.join(residuals_model_results_dir, f'linreg_coefs_split{split_idx}.csv'), index=False)
        standard_scaler_attrs = pd.DataFrame({'feature': pipe.named_steps['scaler'].feature_names_in_, 'scale': pipe.named_steps['scaler'].scale_, 'mean': pipe.named_steps['scaler'].mean_, 'var': pipe.named_steps['scaler'].var_})
        standard_scaler_attrs.to_csv(os.path.join(residuals_model_results_dir, f'standard_scaler_split{split_idx}.csv'), index=False)

        train_df.to_csv(os.path.join(residuals_model_results_dir, f'train_df_split{split_idx}.csv'), index=False)
        test_df.to_csv(os.path.join(residuals_model_results_dir, f'test_df_split{split_idx}.csv'), index=False)

        return train_df, test_df

    def set_data(self, dataset: Dataset):
        super().set_data(dataset)
        self.external_features = hasattr(self.data, 'features') and self.data.features is not None and not self.use_feature_residuals
        if self.use_external_features:
            assert self.external_features, "use_external_features is True, but no external features available in the dataset"

    def run(self):
        # Check if audio_files are available in the dataset
        assert hasattr(self.data, 'audio_files'), "Dataset should have audio_files attribute"

        # Prepare data for regression
        demographics, _ = prepare_demographics(self.data.demographics)

        regression_df = pd.concat([
            self.data.factor_scores_theory.reset_index(drop=True),
            demographics.reset_index(drop=True),  # demographics are necessary for stratified CV
            self.data.mean_composite_cognitive_score.reset_index(drop=True),  # mean composite score, necessary for stratified CV
        ], axis=1)
        regression_df['sample_name'] = np.array(self.data.sample_names)
        regression_df['audio_file'] = np.array(self.data.audio_files)
        regression_df['task'] = np.array(self.data.tasks)

        if self.external_features or self.use_feature_residuals:
            regression_df = pd.concat([regression_df, self.data.features.reset_index(drop=True)], axis=1)
            print(f"Using external features with dimension {self.data.features.shape[1]}")

        # Dropping nan rows
        source_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age', 'mean_composite_cognitive_score']
        disallow_nan_cols = [self.target_variable, 'audio_file'] + source_stratification_cols
        non_nan_filter = ~regression_df[disallow_nan_cols].isna().any(axis=1)
        nan_row_sample_names = regression_df.loc[~non_nan_filter].sample_name
        n_rows_before = regression_df.shape[0]
        if np.sum(~non_nan_filter) > 0:
            print(f"\n\nATTENTION: null values in {np.sum(~non_nan_filter)} rows / {n_rows_before} (sample names {nan_row_sample_names.to_list()})\n{regression_df[disallow_nan_cols].isna().sum()}\n\n")
        regression_df = regression_df.dropna(subset=disallow_nan_cols)
        regression_df = regression_df.reset_index(drop=True)

        assert np.all(regression_df.index == np.arange(regression_df.shape[0]))  # make sure index is reset

        regression_df = resolve_cv_fold_assignment(
            regression_df, self.data, self.CONSTANTS, self.cv_splits,
            data_split_version=self.data_split_version,
            run_parameters=self.run_parameters,
        )

        # Collect results
        scores_test = defaultdict(list)
        scores_dummy = defaultdict(list)
        scores_train = defaultdict(list)

        # Collected data over splits
        sample_names_train_collected, sample_names_test_collected, sample_names_val_collected = [], [], []
        targets_collected, predictions_collected, split_idx_collected = [], [], []
        targets_original_collected = []

        for split_idx in range(self.cv_splits):
            start_time_split = time.time()
            print(f"\n\nRunning split {split_idx} / {self.cv_splits}")

            # Split data
            train_df = regression_df.query(f"test_split != {split_idx}").reset_index(drop=True)
            test_df = regression_df.query(f"test_split == {split_idx}").reset_index(drop=True)

            if self.external_features or self.use_feature_residuals:
                train_df, test_df = self._preprocess_features(train_df, test_df, split_idx)

            if self.use_demographic_residuals or self.use_feature_residuals:
                train_df, test_df = self._prepare_residuals(train_df, test_df, split_idx)

            assert test_df.shape[0] > 0, "Empty test data? Something seems to be wrong with the data splitting"

            if self.cv_splits == 1:
                val_df = train_df.query(f"train_val_split == 'val'").reset_index(drop=True)
                train_df = train_df.query(f"train_val_split == 'train'").reset_index(drop=True)
            else:
                val_df = train_df.query(f'split{split_idx}_val == True').reset_index(drop=True)
                train_df = train_df.query(f'split{split_idx}_val != True').reset_index(drop=True)

            train_df.to_csv(os.path.join(self.run_parameters.results_dir, f"train_df_split{split_idx}.csv"), index=False)
            val_df.to_csv(os.path.join(self.run_parameters.results_dir, f"val_df_split{split_idx}.csv"), index=False)
            test_df.to_csv(os.path.join(self.run_parameters.results_dir, f"test_df_split{split_idx}.csv"), index=False)

            # Prepare datasets
            start_time = time.time()
            train_dataset = self._prepare_dataset(train_df)
            val_dataset = self._prepare_dataset(val_df)
            test_dataset = self._prepare_dataset(test_df)
            store_timing_information(start_time, self.run_parameters.results_dir, f'Prepare datasets')

            # Store sample names
            sample_names_train_collected.append(train_df['sample_name'].tolist())
            sample_names_val_collected.append(val_df['sample_name'].tolist())
            sample_names_test_collected.append(test_df['sample_name'].tolist())

            # Mean predictor (baseline)
            mean_predictor = np.ones(len(test_df)) * train_df[self.target_variable].mean()
            if self.use_demographic_residuals or self.use_feature_residuals:
                mean_predictor = np.zeros(len(test_df))  # training mean of residuals should be 0

            # Train and evaluate model
            print("Training WavLM model")
            start_time = time.time()
            model, y_pred_test, y_true_test, y_pred_train, y_true_train = self._train_and_evaluate(
                train_dataset,
                val_dataset,
                test_dataset,
                split_idx
            )
            store_timing_information(start_time, self.run_parameters.results_dir, f'Train and evaluate model')

            # original target variable (not residuals)
            y_test_original = test_df[self.target_variable].values
            y_train_original = train_df[self.target_variable].values
            test_df[self.target_variable].to_csv(os.path.join(self.run_parameters.results_dir, f"y_test_split{split_idx}.csv"), index=False)
            train_df[self.target_variable].to_csv(os.path.join(self.run_parameters.results_dir, f"y_train_split{split_idx}.csv"), index=False)

            if not (self.use_demographic_residuals or self.use_feature_residuals):
                assert np.all(np.array(y_true_test) - y_test_original < 1e-5)
                assert np.all(np.array(y_true_train) - y_train_original < 1e-5)

            targets_collected.append(y_true_test)
            targets_original_collected.append(y_test_original)
            predictions_collected.append(y_pred_test)
            split_idx_collected.append(np.ones(y_test_original.shape) * split_idx)

            # Compute metrics
            for metric in self.metrics:
                scores_test[metric.__name__].append(metric(y_true_test, y_pred_test))
                scores_dummy[metric.__name__].append(metric(y_true_test, mean_predictor))
                scores_train[metric.__name__].append(metric(y_true_train, y_pred_train))

            if self.only_n_splits is not None and split_idx + 1 >= self.only_n_splits:
                print(f"only_n_splits {self.only_n_splits}, breaking now")
                break

            if self.r2_threshold is not None and split_idx >= 1:
                mean_r2 = np.mean(scores_test['r2_score'])
                if mean_r2 < self.r2_threshold:
                    print(f"R2 threshold {self.r2_threshold} not reached (mean R2 over completed splits {mean_r2}, but need at least {self.r2_threshold}), stopping CV early")
                    break


            store_timing_information(start_time_split, self.run_parameters.results_dir, f"Running split")

        # Log results
        for metric in self.metrics:
            metric_name = metric.__name__
            self._log_results(f"CV {metric_name}: {np.mean(scores_test[metric_name]):.3f}+-{np.std(scores_test[metric_name]):.3f}", 'results.txt')
            self._log_results(f"  (Mean predictor: {np.mean(scores_dummy[metric_name]):.3f}+-{np.std(scores_dummy[metric_name]):.3f})", 'results.txt')
            self._log_results(f"  (Train: {np.mean(scores_train[metric_name]):.3f}+-{np.std(scores_train[metric_name]):.3f})", 'results.txt')

        # Save scores
        with open(os.path.join(self.run_parameters.results_dir, 'scores.json'), "w") as f:
            json.dump({'scores_test': scores_test, 'scores_dummy': scores_dummy, 'scores_train': scores_train}, f)

        # Save sample names
        with open(os.path.join(self.run_parameters.results_dir, 'sample_names.json'), "w") as f:
            json.dump({'sample_names_train': sample_names_train_collected, 'sample_names_test': sample_names_test_collected}, f)

        # Create target and prediction dataframe
        target_and_prediction = pd.DataFrame({
            self.target_variable: np.concatenate(targets_original_collected),
            'target': np.concatenate(targets_collected),
            'prediction': np.concatenate(predictions_collected),
            'split_idx': np.concatenate(split_idx_collected),
            'sample_name': np.concatenate(sample_names_test_collected),
        })

        # Plot results
        plot_target_prediction(target_and_prediction, target_variable=self.target_variable,
                                     fig_store_path=os.path.join(self.run_parameters.results_dir, "prediction_vs_target.png"),
                                     csv_store_path=os.path.join(self.run_parameters.results_dir, "prediction_and_target.csv"))



        # Store data to disk
        target_and_prediction.to_csv(os.path.join(self.run_parameters.results_dir, "regression_data.csv"), index=False)

        return target_and_prediction
