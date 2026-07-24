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
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from transformers import AutoTokenizer, AutoConfig, DebertaV2ForSequenceClassification, DebertaV2Config
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, DebertaV2Model
from transformers.activations import ACT2FN
from datasets import Dataset as HuggingFaceDataset

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





class ContextPooler(nn.Module):
    # Copied from transformers/models/deberta_v2/modeling_deberta_v2.py
    # Fixed the output_dim property to return correct hidden_size
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.pooler_hidden_size


class DebertaV2ForSequenceClassificationFixed(DebertaV2ForSequenceClassification):
    # Fixed the output_dim property to return correct hidden_size
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

class DebertaV2ForSequenceClassificationExtended(DebertaV2ForSequenceClassificationFixed):
    """DebertaV2ForSequenceClassificationExtended, identical to DebertaV2ForSequenceClassification but exposes the pooled_layer (interpreted as features) """
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithFeatures(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            features=pooled_output
        )




class DebertaV2ForSequenceClassificationWithFeatures(DebertaV2ForSequenceClassificationFixed):
    """ With external features, analogously to WavLMWithFeatures """

    def __init__(self, config):
        super().__init__(config)

        # Save external feature dimension --> from config
        try:
            self.external_feature_dim = config.external_feature_dim
        except AttributeError:
            raise AssertionError("external_feature_dim should be specified in config")

        # replace classifier to handle concatenated features
        self.classifier = nn.Linear(self.pooler.output_dim + self.external_feature_dim, self.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        external_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)

        # here the external features are concatenated
        external_features = external_features.to(pooled_output.device, dtype=pooled_output.dtype)
        combined = torch.cat([pooled_output, external_features], dim=-1)

        logits = self.classifier(combined)



        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithFeatures(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            features=combined
        )



class DeBERTa(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__("DeBERTa", *args, **kwargs)

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
                raise NotImplementedError("Percentile Transformation not implemented for WaveLM model")
            else:
                raise ValueError(f"Invalid data preprocessor {p}")


        try:
            self.bert_model_name = self.config.config_model.bert_model_name
        except (AttributeError, KeyError):
            self.bert_model_name = "microsoft/deberta-v3-base"

        try:
            self.batch_size = self.config.config_model.batch_size
        except (AttributeError, KeyError):
            self.batch_size = 2

        try:
            self.learning_rate = float(self.config.config_model.learning_rate)
        except (AttributeError, KeyError):
            self.learning_rate = 5e-5

        try:
            self.hidden_dropout_prob = float(self.config.config_model.hidden_dropout_prob)
        except (AttributeError, KeyError):
            self.hidden_dropout_prob = 0.1  # default in DeBERTa-v3-base

        try:
            self.cls_dropout = float(self.config.config_model.cls_dropout)
        except (AttributeError, KeyError):
            self.cls_dropout = self.hidden_dropout_prob  # default in DeBERTa-v3-base

        try:
            self.pooler_dropout = float(self.config.config_model.pooler_dropout)
        except (AttributeError, KeyError):
            self.pooler_dropout = 0  # default in DeBERTa-v3-base

        try:
            self.num_train_epochs = self.config.config_model.num_train_epochs
        except (AttributeError, KeyError):
            self.num_train_epochs = 60

        # pooler_hidden_size, corresponds to classifier_proj_size in wavlm
        try:
            self.pooler_hidden_size = self.config.config_model.pooler_hidden_size
        except (AttributeError, KeyError):
            self.pooler_hidden_size = 768  # default in DeBERTa-v3-base

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

        print(f"Using target_variable={self.target_variable}, bert_model_name={self.bert_model_name}, "
              f"cv_splits={self.cv_splits}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, "
              f"num_train_epochs={self.num_train_epochs}, "
              f"device={self.device}, hidden_dropout_prob={self.hidden_dropout_prob}, use_lora={self.use_lora}, "
              f"learning_rate_lora={self.learning_rate_lora}, store_models={self.store_models}, "
              f"pooler_hidden_size={self.pooler_hidden_size}, use_external_features={self.use_external_features}"
              f"pretrained_model={self.pretrained_model}, cls_dropout={self.cls_dropout}, "
              f"pooler_dropout={self.pooler_dropout}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)



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

        data_df = data_df.copy()[['transcript', target_variable] + features_cols].rename(columns={target_variable: 'target', 'transcript': 'text'})

        dataset = HuggingFaceDataset.from_pandas(data_df)

        def preprocess(batch):
            tokenized = self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
            tokenized["target"] = batch["target"]
            if self.external_features:
                batch["external_features"] = torch.tensor([batch[c] for c in self.data.features.columns], dtype=torch.float)
            return tokenized

        print("Preprocessing dataset")
        dataset = dataset.map(preprocess, num_proc=1).with_format("torch")  #, num_proc=os.cpu_count()

        return dataset

    def _create_model_and_optimizer(self, split_idx):
        """Create WaveLM model for regression, with LoRA fine-tuning."""

        if self.lora_modules == 'attention_light':
            target_modules = ["query_proj", "value_proj"]
        elif self.lora_modules == 'attention_full':
            target_modules = ["query_proj","key_proj","value_proj","output.dense"]
        else:
            raise ValueError(f"Unknown lora_modules {self.lora_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,  # alpha constant, according to https://github.com/spacepxl/demystifying-sd-finetuning?tab=readme-ov-file, changing alpha is similar to changing learning rate
            lora_dropout=self.lora_dropout, # dropout to apply in the LoRA layers
            r=self.lora_r, # rank controls the capacity of the adaptation, hyperparameter to tune
            target_modules=target_modules,
            bias="none", inference_mode=False, use_rslora=True,
            modules_to_save=["pooler", "classifier"], # keep regression head trainable
        )

        # Note: Default config in '/Users/jheitz/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3/config.json'
        config: DebertaV2Config = AutoConfig.from_pretrained(self.bert_model_name, num_labels=1, problem_type="regression",
                                                             hidden_dropout_prob=self.hidden_dropout_prob,
                                                             cls_dropout=self.cls_dropout,
                                                             pooler_hidden_size=self.pooler_hidden_size,
                                                             pooler_dropout=self.pooler_dropout)
        if self.external_features and self.use_external_features:
            config.external_feature_dim = self.data.features.shape[1]
            RegressionClass = DebertaV2ForSequenceClassificationWithFeatures
        else:
            RegressionClass = DebertaV2ForSequenceClassificationExtended

        deberta_regression: DebertaV2ForSequenceClassification = RegressionClass.from_pretrained(self.bert_model_name, config=config)

        if self.use_lora:
            model = get_peft_model(deberta_regression, lora_config)

            # LoRA actually injected?
            injected = [n for n, _ in model.named_parameters() if "lora_" in n]
            print("LoRA params:", len(injected));
            assert injected, "No LoRA params — target_modules mismatch."

            # function to store trained parameters (just relevant ones)
            model_save_fct = lambda save_dir: model.save_pretrained(save_dir, safe_serialization=True)

            # model to restore from stored parameters
            def model_restore_fct(restore_dir):
                deberta_regression = RegressionClass.from_pretrained(self.bert_model_name, config=config)
                model = PeftModel.from_pretrained(deberta_regression, restore_dir)
                model.to(self.device)
                return model

        else:
            model = deberta_regression

            # function to store trained parameters (just relevant ones)
            model_save_fct = lambda save_dir: model.save_pretrained(save_dir, safe_serialization=True)

            # model to restore from stored parameters
            model_restore_fct = lambda restore_dir: RegressionClass.from_pretrained(restore_dir).to(self.device)

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
                                                   or "pooler.modules_to_save" in n)]

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
        """Train and evaluate WaveLM model."""
        # Create model
        model, optimizer, model_save_fct, model_restore_fct = self._create_model_and_optimizer(split_idx)

        loss_fn = nn.MSELoss()

        def predict(model, batch):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device).float().view(-1)
            external_features = batch["external_features"].to(self.device) if self.external_features else None
            if self.use_external_features:
                model_output = model(input_ids=input_ids, attention_mask=attention_mask, external_features=external_features)
                feature_values = model_output.features
            else:
                model_output = model(input_ids=input_ids, attention_mask=attention_mask)
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
        regression_df['transcript'] = np.array(self.data.transcripts)
        regression_df['task'] = np.array(self.data.tasks)

        if self.external_features or self.use_feature_residuals:
            regression_df = pd.concat([regression_df, self.data.features.reset_index(drop=True)], axis=1)
            print(f"Using external features with dimension {self.data.features.shape[1]}")

        # Dropping nan rows
        source_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age', 'mean_composite_cognitive_score']
        disallow_nan_cols = [self.target_variable, 'transcript'] + source_stratification_cols
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
                mean_predictor = np.zeros(len(test_df))  # training mean of residuals should be 0?

            # Train and evaluate model
            print(f"Training {self.bert_model_name} model")
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
