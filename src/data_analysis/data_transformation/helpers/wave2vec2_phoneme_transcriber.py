import torch
import numpy as np
import pandas as pd
from transformers import pipeline
import time

from data_analysis.dataloader.dataset import Dataset
from data_analysis.util.decorators import cache_to_file_decorator
from data_analysis.data_transformation.data_transformer import DataTransformer
from util.helpers import hash_from_dict, store_timing_information


class Wave2Vec2PhonemeTranscriber(DataTransformer):
    def __init__(self, *args, **kwargs):
        self.name = "wav2vec2-xls-r-300m-timit-phoneme"
        super().__init__(*args, **kwargs)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("... running Wave2Vec2PhonemeTranscriber on device: {}".format(self.device))
        model_kwargs = dict(cache_dir=self.CONSTANTS.CACHE_DIR) if not self.CONSTANTS.science_cluster else None
        self.pipe = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme", device=self.device,
                             model_kwargs=model_kwargs)

        self.transcriber_config = {}  # any config for the transcriber = different versions
        self.version = 1  # version of the transcriber's code -> if significant logic changes, change this

    @property
    def _version_config(self):
        # a short string representing the version and config hash.
        # this should make a particular version of the code and config of the transcriber unique
        # used for saving transcriptions to files and handling the caches
        config_hash = hash_from_dict(self.transcriber_config, 6)
        version_config = f"v{self.version}_{config_hash}"
        return version_config

    @cache_to_file_decorator(n_days=60)
    def transcribe_file(self, file_path: str, sample_name: str, version_config: str) -> str:
        if file_path is None or pd.isna(file_path):
            return None

        print(f"Transcribing sample_name {sample_name} file {file_path}", end="... ")

        start_time = time.time()
        transcription = self.pipe(file_path, chunk_length_s=10, stride_length_s=(4, 2))['text']
        store_timing_information(start_time, self.run_parameters.results_dir, f"phoneme transcription")

        print(f"{transcription[:20]}...")

        return transcription

    def transcribe_dataset(self, dataset: Dataset) -> Dataset:
        transcribed_data = np.array(
            [self.transcribe_file(file, sample_name=sample_name, version_config=self._version_config)
             for file, sample_name in zip(dataset.audio_files, dataset.sample_names)]
        )
        return transcribed_data
