import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset
from util.helpers import prepare_demographics

class TaskFeatures(DataTransformer):
    """
    Add task as a feature (useful when all_individual, s.t. potential task-level biases are captured)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "TaskFeatures"
        print(f"Initializing {self.name}")
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')

    def _load_features(self, dataset: Dataset):
        assert isinstance(dataset, Dataset), "Input should be Dataset (manual transcripts)"

        tasks = pd.DataFrame({'task': dataset.tasks})
        tasks_onehot = self.one_hot_encoder.fit_transform(tasks)
        features_df = pd.DataFrame(tasks_onehot, columns=self.one_hot_encoder.get_feature_names_out())

        config_without_transformers = {key: dataset.config[key] for key in dataset.config if key != 'data_transformers'}
        new_config = {
            'data_transformers': [*dataset.config['data_transformers'], self.name],
            **config_without_transformers
        }

        new_dataset = dataset.copy()
        new_dataset.name = f"{dataset.name} - {self.name}"
        new_dataset.config = new_config

        new_dataset.features = self._create_new_feature_df(new_dataset, features_df)

        return new_dataset

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print(f"Collecting task feature for dataset {dataset}")
        return self._load_features(dataset)





