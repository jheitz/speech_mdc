import os

import numpy as np
import pandas as pd
from pydub import AudioSegment
import functools

from config.config import Config
from config.constants import Constants
from data_analysis.dataloader.dataset import Dataset, DatasetType
from data_analysis.util.decorators import cache_to_file_decorator
from util.helpers import hash_list, prepare_demographics
from sklearn.model_selection import train_test_split

from data_analysis.dataloader.ADReSS_logic.dataloader import ADReSSWithPITTDataLoader as OriginalADReSSWithPITTDataLoader
from data_analysis.dataloader.ADReSS_logic.audio_cutter import AudioCutter
from util.google_speech_transcription import GoogleSpeechTranscriber

class LuhaBaseDataLoader:
    """
    LUHA dataloader
    The logic that is common in the 2024 and 2026 version is kept here
    """
    def __init__(self, debug=False, local=None, constants=None, config=None, run_parameters=None, name="LUHA dataloader", dataset_type=None):
        self.debug = debug
        self.name = name
        if constants is None:
            self.CONSTANTS = Constants(local=local)
        else:
            self.CONSTANTS = constants
        self.config = config
        self.run_parameters = run_parameters

        self.dataset_type = dataset_type
        assert self.dataset_type in [DatasetType.LUHA2024, DatasetType.LUHA2026, DatasetType.ADReSS, DatasetType.LUHACombined], f"Invalid dataset type {dataset_type}"

        try:
            self.transcript_version = self.config.config_data.transcript_version
        except (AttributeError, KeyError):
            self.transcript_version = 'google'
        valid_transcript_versions = ['google', 'whisper']
        assert self.transcript_version in valid_transcript_versions, f"Invalid transcript version {self.transcript_version}. Should be in {valid_transcript_versions}"

        try:
            self.task = self.config.config_data.task
        except (AttributeError, KeyError):
            self.task = 'pictureDescription'
            print(f"No task specified, loading {self.task}")
        if self.dataset_type == DatasetType.LUHA2024:
            valid_tasks = ['cookieTheft', 'journaling', 'phonemicFluency', 'picnicScene', 'semanticFluency', 'pictureDescription', 'pictureNaming', 'all_individual']
        elif self.dataset_type == DatasetType.LUHA2026:
            valid_tasks = ['cookieTheft', 'journaling', 'pictureDescription', 'pictureNaming', 'cookieTheftImmediateRecall', 'cookieTheftDelayedRecall', 'picnicScene', 'all_individual']
        elif self.dataset_type == DatasetType.LUHACombined:
            valid_tasks = ['cookieTheft', 'journaling', 'picnicScene', 'pictureDescription', 'pictureNaming', 'all_individual']
        else:
            valid_tasks = []
        assert not isinstance(self.task, list), "Task is a list. This can be a valid config for run_multiple.py, which runs multiple setups with different tasks. Here (run.py / main.py), you can only have one task"
        assert self.task in valid_tasks, f"Invalid task {self.task}. Should be in {valid_tasks}"

        try:
            self.consent_filter = self.config.config_data.consent_filter
        except (AttributeError, KeyError):
            self.consent_filter = 'full'  # full dataset
            print(f"No consent_filter specified (consent for further data use), loading all participants")
        consent_filters = ['full', 'futher_use_with_audio', 'further_use_without_audio']
        assert self.consent_filter in consent_filters, f"Invalid transcript version {self.consent_filter}. Should be in {consent_filters}"

        try:
            self.store_loaded_dataset = self.config.config_data.store_loaded_dataset
        except:
            self.store_loaded_dataset = False

        try:
            self.waves = self.config.config_data.waves
        except (AttributeError, KeyError):
            self.waves = None
        assert self.waves is None or all([w in ['wave1', 'wave2', 'wave3'] for w in self.waves]), f"Invalid waves: {self.waves}"

        waves_and_ids = pd.read_csv(self.CONSTANTS.WAVES_AND_IDS, dtype=str)
        self.waves_and_ids = waves_and_ids.rename(columns={'longitudinal_id': 'sample_name_longitudinal', 'study_submission_id': 'sample_name'})


    def _preprocess_pandas_data(self, df, sample_names):
        """
        Preprocess a dataframe with sample-level data
        - replace study_submission_id by sample_name, if present
        - sort by external sample_names list
        """
        assert len(set(sample_names)) == len(sample_names), f"Cannot use this function with sample_names that contain duplicates."
        sample_names = list(sample_names)
        df = df.copy().rename(columns={"study_submission_id": "sample_name"})
        df['sample_name'] = df['sample_name'].astype(str)
        assert len(sample_names) == df.shape[0], f"{len(sample_names)} vs. {df.shape[0]}: Diff: {[s for s in sample_names if s not in df.sample_name.to_list()]} / {[s for s in df.sample_name if s not in sample_names]}"
        if self.task != 'all_individual':
            assert len(set(sample_names)) == len(sample_names), f"Duplicate sample names: {set(sample_names)}"
        assert all([s in sample_names for s in df['sample_name']]), f"Sample name {df['sample_name'][~df['sample_name'].isin(sample_names)].to_list()} in df but not in sample_names list"
        return df.sort_values(by="sample_name", key=lambda sn: sn.map(lambda e: sample_names.index(e))).reset_index(drop=True)

    def _prepare_cognitive_overall_score(self, acs_scores, language_task_scores):
        df = pd.merge(acs_scores, language_task_scores, how='outer', on="sample_name").copy()
        if self.dataset_type == DatasetType.LUHA2026:
            df['phonemic_fluency_score'] = df['phonemic_fluency_f_score'] + df['phonemic_fluency_a_score'] + df['phonemic_fluency_s_score']
            df['semantic_fluency_score'] = df['semantic_fluency_score'] + df['semantic_fluency_vegetables_score']
        language_task_score_names = ['phonemic_fluency_score', 'semantic_fluency_score', 'picture_naming_score']
        cognitive_cols = self.CONSTANTS.ACS_MAIN_OUTCOME_VARIABLES + language_task_score_names
        for col in cognitive_cols:
            df[col] = (df.loc[:, col] - df.loc[:, col].mean()) / df.loc[:, col].std()  # standardize cols
        mean = df[cognitive_cols].mean(axis=1)
        return pd.DataFrame({'cognition_mean': mean, 'sample_name': df.sample_name})

    def _split_transcripts(self, transcriptions_df, version):
        assert version in ['google', 'whisper']
        relevant_cols = ['task', 'sample_name', f"text_{version}"]
        df_version = transcriptions_df[relevant_cols].rename(columns={f'text_{version}': 'transcription'})
        df_version_cleaned = df_version[~df_version.task.str.contains("check")]
        pivoted = df_version_cleaned.pivot(index='sample_name', columns='task', values="transcription").reset_index()
        return pivoted

    @cache_to_file_decorator(n_days=10, verbose=False)
    def _concatenate_audio_files(self, list_of_audio_paths):
        # concatenate multiple audio files into one new audio file (with 1sec pause in between)
        # the new file is written to the cache directory

        sounds = [AudioSegment.from_file(path, format="wav") for path in list_of_audio_paths]
        concatenated = functools.reduce(lambda sound1, sound2: sound1 + AudioSegment.silent(duration=1000) + sound2, sounds)

        basenames = [os.path.basename(p).replace(".wav", "") for p in list_of_audio_paths]
        hash = hash_list(list_of_audio_paths, hash_len=10)
        new_basename = "+".join(basenames) + "_" + hash + ".wav"
        output_dir = os.path.join(self.CONSTANTS.CACHE_DIR_CENTRALIZED, "concatenated_audio")
        os.makedirs(output_dir, exist_ok=True)
        new_path = os.path.join(output_dir, new_basename)

        with open(new_path, "wb") as f:
            concatenated.export(f, format="wav")

        return new_path

    def _load_extra_targets(self, acs_outcomes_raw):
        extra_targets = pd.DataFrame({
            f"{col}_standardized": (acs_outcomes_raw[col] - acs_outcomes_raw[col].mean()) / acs_outcomes_raw[col].std() for col in [c for c in acs_outcomes_raw.columns if 'questionnaire' in c]
        })
        return extra_targets

    def _load_acs_questionnaires(self, acs_outcomes_raw):
        acs_questionnaires = acs_outcomes_raw[[c for c in acs_outcomes_raw.columns if 'questionnaire' in c] + ['sample_name']]
        return acs_questionnaires

    def _load_data_common(self):
        audio_files = []
        sample_names = []
        transcriptions = []
        for i, file_name in enumerate(sorted(os.listdir(self.processed_data_dir), key=lambda id_str: id_str.zfill(4))):
            if os.path.isdir(os.path.join(self.processed_data_dir, file_name)):
                sample_dir = os.path.join(self.processed_data_dir, file_name)
                if self.dataset_type == DatasetType.LUHA2024:
                    sample_audio_dir = sample_dir
                elif self.dataset_type == DatasetType.LUHA2026:
                    sample_audio_dir = os.path.join(sample_dir, "audio")
                else: raise ValueError(f"Invalid dataset type {self.dataset_type}")
                sample_name = file_name
                sample_audio_files = {'sample_name': sample_name}
                for j, file_name in enumerate(list(os.listdir(sample_audio_dir))):
                    if file_name.endswith('.wav'):
                        if 'check' in file_name:
                            continue  # ignore check files
                        task_name = file_name.replace(".wav", "")
                        file_path = os.path.join(sample_audio_dir, file_name)
                        assert task_name not in sample_audio_files, \
                            f"Duplicate task {task_name} for sample {sample_name}?"
                        sample_audio_files[task_name] = file_path
                audio_files.append(sample_audio_files)
                sample_names.append(sample_name)

                sample_transcriptions = pd.read_csv(os.path.join(sample_audio_dir, "ASR", "transcriptions.csv"))
                sample_transcriptions = sample_transcriptions.rename(columns={'submission': 'sample_name'})
                transcriptions.append(sample_transcriptions)

        audio_files_df = pd.DataFrame(audio_files)

        transcriptions_df = pd.concat(transcriptions)
        transcriptions_google = self._split_transcripts(transcriptions_df, 'google')
        transcriptions_google = self._preprocess_pandas_data(transcriptions_google, sample_names)
        transcriptions_whisper = self._split_transcripts(transcriptions_df, 'whisper')
        transcriptions_whisper = self._preprocess_pandas_data(transcriptions_whisper, sample_names)

        acs_outcomes_raw = pd.read_csv(os.path.join(self.processed_data_dir, "acs_outcomes_raw.csv"))
        acs_outcomes_raw = self._preprocess_pandas_data(acs_outcomes_raw, sample_names)
        acs_outcomes_imputed = pd.read_csv(os.path.join(self.processed_data_dir, "acs_outcomes_imputed.csv"), dtype={'study_submission_id': int})
        acs_outcomes_imputed = self._preprocess_pandas_data(acs_outcomes_imputed, sample_names)
        demographics = pd.read_csv(os.path.join(self.processed_data_dir, "demographics.csv"),
                           dtype={'2024_study_submission_id': str})
        demographics = self._preprocess_pandas_data(demographics, sample_names)
        language_task_scores = pd.read_csv(os.path.join(self.processed_data_dir, "language_task_scores.csv"))
        language_task_scores = self._preprocess_pandas_data(language_task_scores, sample_names)

        if self.transcript_version == 'google':
            transcripts = transcriptions_google
        elif self.transcript_version == 'whisper':
            transcripts = transcriptions_whisper
        else:
            raise ValueError(f"Invalid transcript version {self.transcript_version}")

        if self.task == 'pictureDescription':
            # combine cookieTheft and picnicScene transcripts
            print("Combining cookieTheft and picnicScene to get pictureDescription...", end=" ")
            def combine_cols_transcripts(row):
                if pd.isna(row['cookieTheft']) or pd.isna(row['picnicScene']):
                    return None
                return f"{row.cookieTheft}\n {row.picnicScene}"
            transcripts_numpy = transcripts.apply(combine_cols_transcripts, axis=1).to_numpy()
            def combine_cols_audio(row):
                if pd.isna(row['cookieTheft']) or pd.isna(row['picnicScene']):
                    return None
                return self._concatenate_audio_files([row['cookieTheft'], row['picnicScene']])
            audio_files_for_task = audio_files_df.apply(combine_cols_audio, axis=1).to_numpy()
            print("... done.")
        elif self.task == 'all_individual':
            transcripts_numpy = np.array([np.nan for _ in range(len(sample_names))])
            audio_files_for_task = np.array([np.nan for _ in range(len(sample_names))])
        else:
            transcripts_numpy = transcripts[self.task].to_numpy()
            audio_files_for_task = audio_files_df[self.task].to_numpy()

        cognitive_overall_score = self._prepare_cognitive_overall_score(acs_outcomes_imputed, language_task_scores)
        cognitive_overall_score =  self._preprocess_pandas_data(cognitive_overall_score, sample_names)

        extra_targets = self._load_extra_targets(acs_outcomes_raw)
        acs_questionnaires = self._load_acs_questionnaires(acs_outcomes_raw)
        acs_questionnaires = self._preprocess_pandas_data(acs_questionnaires, sample_names)

        sample_names_longitudinal_mapping = pd.DataFrame({'sample_name': sample_names}).merge(self.waves_and_ids, on='sample_name', how='left')
        assert np.all(sample_names_longitudinal_mapping['sample_name'] == sample_names), "Sample name order should not change"
        waves = sample_names_longitudinal_mapping.wave.to_numpy()
        sample_names_longitudinal = sample_names_longitudinal_mapping.sample_name_longitudinal.to_numpy()

        return sample_names, sample_names_longitudinal, audio_files_for_task, audio_files_df, transcriptions_google, transcriptions_whisper, transcripts_numpy, transcripts, \
            acs_outcomes_raw, acs_outcomes_imputed, demographics, language_task_scores, cognitive_overall_score, \
            extra_targets, acs_questionnaires, waves


    def _calculate_mean_composite_cognitive_score(self, factor_scores_theory, sample_names):
        mean_composite_cognitive_score = pd.DataFrame({
            'mean_composite_cognitive_score': factor_scores_theory[[c for c in factor_scores_theory.columns if 'composite_' in c]].mean(axis=1),
            'sample_name': factor_scores_theory.sample_name
        })
        mean_composite_cognitive_score =  self._preprocess_pandas_data(mean_composite_cognitive_score, sample_names)
        return mean_composite_cognitive_score

    def _apply_consent_filter(self, dataset):
        if self.consent_filter != 'full':
            # filter out samples that do not have the consent filter
            if self.consent_filter == 'futher_use_with_audio':
                sample_names = dataset.sample_names[dataset.demographics.consent_data_further_use == "Yes"]
            elif self.consent_filter == 'further_use_without_audio':
                sample_names = dataset.sample_names[dataset.demographics.consent_data_further_use.isin(["Yes", "Yes Except Recordings"])]
            else:
                raise ValueError(f"Invalid consent filter {self.consent_filter}")
            dataset = dataset.subset_from_sample_names(sample_names)
            dataset.name = f"{dataset.name} - Consent Filter {self.data_split}"
        return dataset

    @staticmethod
    def _align_dataframe_columns(dfs):
        """Return (df1, df2) trimmed to the intersection of columns, preserving df1's column order."""
        common_cols = [c for c in dfs[0].columns if c in set(set.intersection(*map(set, [df.columns for df in dfs])))]
        return [df[common_cols] for df in dfs]

    def _multiply_dataset_with_individual_spontaneous_speech_tasks(self, dataset, transcripts, audio_files_df):
        print("Combining all spontaneous speech tasks as individual samples")
        datasets = []
        for task in ['cookieTheft', 'journaling', 'picnicScene']:
            dataset_here = dataset.copy()
            dataset_here.transcripts = transcripts[task].to_numpy()
            dataset_here.audio_files = audio_files_df[task].to_numpy()
            dataset_here.tasks = np.array([task] * len(dataset))
            datasets.append(dataset_here)

        # Determine common data variables (sample_names handled separately as a positional arg)
        dataset_vars = [set(dataset.data_variables.keys()) for dataset in datasets]
        common_vars = sorted(set.intersection(*dataset_vars) - {'sample_names'})

        combined_sample_names = np.concatenate([dataset.sample_names for dataset in datasets])

        combined_data = {}
        for var in common_vars:
            dataset_vals = [dataset.data_variables[var] for dataset in datasets]

            if all(isinstance(val, pd.DataFrame) for val in dataset_vals):
                dataset_vals = self._align_dataframe_columns(dataset_vals)
                vs = []

                # Re-attach sample_name so Dataset constructor can validate ordering
                for dataset, val in zip(datasets, dataset_vals):
                    v = val.copy()
                    v['sample_name'] = dataset.sample_names
                    vs.append(v)
                combined_data[var] = pd.concat(vs, axis=0).reset_index(drop=True)
            elif all(isinstance(val, np.ndarray) for val in dataset_vals):
                combined_data[var] = np.concatenate(dataset_vals, axis=0)
            else:
                raise TypeError(f"Cannot combine {var}: types {[type(val for val in dataset_vals)]}")

        combined_dataset = Dataset(
            name=dataset.name,
            type=dataset.type,
            sample_names=combined_sample_names,
            config=dataset.config,
            **combined_data,
        )
        return combined_dataset

    def load_data(self):
        raise NotImplementedError("Abstract method")


class Luha2024DataLoader(LuhaBaseDataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'Luha 2024 Dataloader'
        kwargs['dataset_type'] = DatasetType.LUHA2024
        super().__init__(*args, **kwargs)

        try:
            self.data_split = str(self.config.config_data.split)
        except (AttributeError, KeyError):
            self.data_split = 'train'
            print(f"No data split specified, loading train participants")
        valid_data_split = ['train', 'test', 'full', '1', '2', 'n100', 'n400', 'returning_2026', 'not_returning_2026']
        assert self.data_split in valid_data_split or os.path.exists(self.data_split), f"Invalid split {self.data_split}. Should be in {valid_data_split} or a valid path"

        if self.waves is not None:
            assert self.data_split == 'full', f"Either use data_split or waves, not both. waves = ['wave1', **] corresponds to data_split = 'full' in the 2024 data. Now you have waves={self.waves}, data_split={self.data_split}"

        self.processed_data_dir = os.path.join(self.CONSTANTS.DATA_PROCESSED_COMBINED, "data")

        consent_filter_text = f", {self.consent_filter}" if self.consent_filter != 'full' else ""
        split_text = f" Split {self.data_split}" if self.data_split is not None else ""
        print(f"Initializing dataloader {self.name} (transcript_version {self.transcript_version}, task {self.task}{consent_filter_text}){split_text}")

    def load_data(self):
        print(f"Loading data using dataloader {self.name}")

        sample_names, sample_names_longitudinal, audio_files_for_task, audio_files_df, transcriptions_google, transcriptions_whisper, transcripts_numpy, transcripts, \
            acs_outcomes_raw, acs_outcomes_imputed, demographics, language_task_scores, cognitive_overall_score, \
            extra_targets, acs_questionnaires, waves = self._load_data_common()

        assert np.all(waves == 'wave1'), f"All 2024 data should be wave1, but there are: {np.Series(waves).value_counts()}"

        # versions of factor scores / composite scores: check the /factor_analysis/R/2_CFA_theory-driven.R for the logic
        # @deprecated: kept for backward compatibility for the moment
        try:
            factor_scores_theory_version = self.config.config_data.factor_scores_theory_version
            assert factor_scores_theory_version == '2025-01-07-1929', "Only factor_scores_theory_version 2025-01-07-1929 is supported."
        except:
            factor_scores_theory_version = None
        factor_scores_theory = pd.read_csv(self.CONSTANTS.FACTOR_SCORES_THEORY_CSV)

        print("Standardizing theory factor scores to zero mean and unit variance. This makes comparison of certain evaluation metrics, beta coefficients etc. more interpretable.")
        composite_score_columns = [c for c in factor_scores_theory.columns if c != 'study_submission_id']
        factor_scores_theory[composite_score_columns] = factor_scores_theory[composite_score_columns].apply(lambda col: (col - np.mean(col)) / np.std(col))
        factor_scores_theory = factor_scores_theory.rename(columns={c: f"composite_{c}" for c in composite_score_columns})
        factor_scores_theory = self._preprocess_pandas_data(factor_scores_theory, sample_names)

        mean_composite_cognitive_score = self._calculate_mean_composite_cognitive_score(factor_scores_theory, sample_names)


        dataset_name = f"LUHA 2024 data ({self.transcript_version} {self.task})"
        dataset = Dataset(name=dataset_name, type=DatasetType.LUHA2024, sample_names=np.array(sample_names), sample_names_longitudinal=sample_names_longitudinal,
                          config={'data_transformers': [], 'debug': self.debug, 'data_split': self.data_split},
                          audio_files=audio_files_for_task, waves=waves,
                          acs_outcomes_imputed=acs_outcomes_imputed, acs_outcomes_raw=acs_outcomes_raw,
                          demographics=demographics, language_task_scores=language_task_scores,
                          factor_scores_theory=factor_scores_theory,
                          cognitive_overall_score=cognitive_overall_score, transcriptions_google=transcriptions_google,
                          transcriptions_whisper=transcriptions_whisper, transcripts=transcripts_numpy,
                          mean_composite_cognitive_score=mean_composite_cognitive_score,
                          extra_targets=extra_targets, acs_questionnaires=acs_questionnaires,
                          tasks=np.array([self.task] * len(sample_names)),
                          )


        if self.task == 'all_individual':
            dataset = self._multiply_dataset_with_individual_spontaneous_speech_tasks(dataset, transcripts, audio_files_df)


        if self.data_split == 'full':
            # the full data should only be used in a train/test setting, not for cross-validation
            try:
                cv_splits = self.config.config_model.cv_splits
            except:
                cv_splits = None
            assert cv_splits == 1, f"Full dataset should only be used in a train/test setup, but cv_splits={cv_splits}"

        else:
            if self.data_split in ['1', '2']:
                # Data split 1 or 2. This is a split in two stratified halves of the data
                # For the logic, check /analyses/kw26/data_split_for_factor_analysis.ipynb
                assignment = pd.read_csv(self.CONSTANTS.SPLIT1_SPLIT2_DATASPLIT)
                # Remove two submissions that are no longer part of the dataset (due to missing scores, decided after calculating the data split)
                assignment = assignment[~assignment.study_submission_id.isin([1279, 1303, 1333])]
                sample_names = assignment[assignment.split.astype(str) == self.data_split].study_submission_id
            elif self.data_split in ['train', 'test']:
                # This is a newer data split into train and test (80% / 20%)
                # Train should be used for training and model selection, test only once at the end for reporting
                # For the logic, check /analyses/kw47/train_test_split.ipynb
                assignment = pd.read_csv(self.CONSTANTS.TRAIN_TEST_DATASPLIT)
                sample_names = assignment[assignment.split == self.data_split].study_submission_id
            elif self.data_split == 'returning_2026':
                longitudinal_ids = pd.read_csv(self.CONSTANTS.LONGITUDINAL_PARTICIPANT_IDS_2024_2026, dtype='str')
                longitudinal_sample_names_2024 = longitudinal_ids['2024_study_submission_id'].to_list()
                sample_names = set(longitudinal_sample_names_2024)
            elif self.data_split == 'not_returning_2026':
                longitudinal_ids = pd.read_csv(self.CONSTANTS.LONGITUDINAL_PARTICIPANT_IDS_2024_2026, dtype='str')
                longitudinal_sample_names_2024 = longitudinal_ids['2024_study_submission_id'].to_list()
                sample_names = list(set(sample_names) - set(longitudinal_sample_names_2024))
            elif os.path.exists(self.data_split):
                # custom data split from a file
                assignment = pd.read_csv(self.data_split)
                assert 'sample_name' in assignment.columns, f"Custom data split file {self.data_split} should contain a sample_name column"
                sample_names = assignment['sample_name']
            else:
                raise ValueError()

            dataset = dataset.subset_from_sample_names(sample_names)
            dataset.name = f"{dataset_name} - Split {self.data_split}"

        dataset = self._apply_consent_filter(dataset)


        if self.task == 'cookieTheft':
            # remove persons with empty transcript features for cookieTheft task, see analysis in /analyses/kw48/find_outlier_participants.ipynb
            empty_transcripts_sample_names = ['98', '138', '141', '469', '516', '1065']
            non_empty_transcript_filter = [s for s in dataset.sample_names if s not in empty_transcripts_sample_names]
            print(f"\n\nATTENTION: removing {len(dataset.sample_names) - len(non_empty_transcript_filter)} participants with empty transcript on the cookieTheft task (sample names {empty_transcripts_sample_names})\n\n")
            dataset = dataset.subset_from_sample_names(non_empty_transcript_filter)
            dataset.name = f"{dataset_name} - Removed participants without cookieTheft transcript"
        elif self.task in ['picnicScene', 'journaling']:
            transcripts_df = pd.DataFrame({'sample_name': dataset.sample_names, 'transcript': dataset.transcripts})
            transcripts_df['transcript'] = transcripts_df['transcript'].apply(lambda t: len(t) if not pd.isna(t) else 0).sort_values(ascending=True)
            non_empty_transcript_filter = transcripts_df[transcripts_df['transcript'] > 100].sample_name.to_list()
            empty_transcripts_sample_names = [s for s in dataset.sample_names if s not in non_empty_transcript_filter]
            print(f"\n\nATTENTION: removing {len(dataset.sample_names) - len(non_empty_transcript_filter)} participants with empty transcript on the {self.task} task (sample names {empty_transcripts_sample_names})\n\n")
            dataset = dataset.subset_from_sample_names(non_empty_transcript_filter)
            dataset.name = f"{dataset_name} - Removed participants without {self.task} transcript"

        if self.waves is not None:
            wave_filter = np.isin(dataset.waves, self.waves)
            dataset = dataset.subset_from_indices(list(np.where(wave_filter)[0]))
            dataset.name = f"{dataset_name} - Waves={self.waves}"
            assert np.all(np.isin(dataset.waves, self.waves))

        if self.debug:
            n_samples = self.debug if self.debug > 1 else 10
            dataset = dataset.subset_from_indices(list(range(n_samples)))

        if self.store_loaded_dataset:
            dataset.store_to_disk(os.path.join(self.run_parameters.results_dir, "dataset_after_dataloader"))

        return dataset





class Luha2026DataLoader(LuhaBaseDataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'Luha 2026 Dataloader'
        kwargs['dataset_type'] = DatasetType.LUHA2026
        super().__init__(*args, **kwargs)

        self.processed_data_dir = os.path.join(self.CONSTANTS.DATA_PROCESSED_COMBINED_2026, "data")

        consent_filter_text = f", {self.consent_filter}" if self.consent_filter != 'full' else ""
        print(f"Initializing dataloader {self.name} (transcript_version {self.transcript_version}, task {self.task}{consent_filter_text})")


    def _inclusion_filter(self, list_like, inclusion_filter):
        if isinstance(list_like, pd.DataFrame):
            return list_like[inclusion_filter].reset_index(drop=True)
        elif isinstance(list_like, pd.Series):
            return list_like[inclusion_filter].reset_index(drop=True)
        elif isinstance(list_like, np.ndarray):
            return list_like[inclusion_filter]
        elif isinstance(list_like, list):
            return [item for item, include in zip(list_like, inclusion_filter) if include]
        else:
            raise ValueError(f"Unsupported type for inclusion_filter: {type(list_like)}")

    def load_data(self):
        print(f"Loading data using dataloader {self.name}")

        sample_names, sample_names_longitudinal, audio_files_for_task, audio_files_df, transcriptions_google, transcriptions_whisper, transcripts_numpy, transcripts, \
            acs_outcomes_raw, acs_outcomes_imputed, demographics, language_task_scores, cognitive_overall_score, \
            extra_targets, acs_questionnaires, waves = self._load_data_common()

        assert np.all(pd.Series(waves).isin(['wave2', 'wave3'])), f"2026 data should be ['wave2', 'wave3'], but there are: {np.Series(waves).value_counts()}"

        demographics = demographics.rename(columns={"2024_study_submission_id": "2024_sample_name"})
        demographics['longitudinal'] = demographics['2024_sample_name'].apply(lambda x: 1 if pd.notna(x) else 0)

        # moca scores
        moca_scores = pd.read_csv(os.path.join(self.processed_data_dir, "moca_scores.csv"))
        moca_scores['total_unknown_score'] = 30 - moca_scores['total_score'] - moca_scores['total_missed_score']
        moca_scores = self._preprocess_pandas_data(moca_scores, sample_names)

        # versions of factor scores / composite scores: check the /factor_analysis/R/2_CFA_theory-driven.R for the logic
        try:
            factor_scores_theory_version = self.config.config_data.factor_scores_theory_version
        except:
            factor_scores_theory_version = None
        if factor_scores_theory_version is not None:
            raise NotImplementedError("Factor scores theory version is not used anymore, use compatible_wiith_2024=True or False to choose the correct version")

        try:
            compatible_with_2024 = self.config.config_data.compatible_with_2024
        except:
            compatible_with_2024 = False
        factor_scores_theory_csv = self.CONSTANTS.FACTOR_SCORES_THEORY_2026_COMPATIBLE_WITH_2024 if compatible_with_2024 else self.CONSTANTS.FACTOR_SCORES_THEORY_2026_INDEPENDENT
        print(f"Using compatible_with_2024={compatible_with_2024}, i.e. factor scores at {factor_scores_theory_csv}")
        factor_scores_theory = pd.read_csv(factor_scores_theory_csv)

        if compatible_with_2024:
            print("Standardizing theory factor scores according to 2024 mean and std")
            print(f"... Using 2024 theory factor scores at {self.CONSTANTS.FACTOR_SCORES_THEORY_CSV} for standardization")
            factor_scores_theory_2024 = pd.read_csv(self.CONSTANTS.FACTOR_SCORES_THEORY_CSV)
            composite_score_columns = [c for c in factor_scores_theory.columns if c != 'study_submission_id']
            for col in composite_score_columns:
                factor_scores_theory[col] = (factor_scores_theory[col] - factor_scores_theory_2024[col].mean()) / factor_scores_theory_2024[col].std()
        else:
            print("Standardizing theory factor scores to zero mean and unit variance.")
            composite_score_columns = [c for c in factor_scores_theory.columns if c != 'study_submission_id']
            factor_scores_theory[composite_score_columns] = factor_scores_theory[composite_score_columns].apply(lambda col: (col - np.mean(col)) / np.std(col))

        factor_scores_theory = factor_scores_theory.rename(columns={c: f"composite_{c}" for c in composite_score_columns})
        factor_scores_theory = self._preprocess_pandas_data(factor_scores_theory, sample_names)

        mean_composite_cognitive_score = self._calculate_mean_composite_cognitive_score(factor_scores_theory, sample_names)

        dataset_name = f"LUHA 2026 data ({self.transcript_version} {self.task})"
        dataset = Dataset(name=dataset_name, type=DatasetType.LUHA2026, sample_names=np.array(sample_names), sample_names_longitudinal=sample_names_longitudinal,
                          config={'data_transformers': [], 'debug': self.debug, 'data_split': None},
                          audio_files=audio_files_for_task, waves=waves,
                          acs_outcomes_imputed=acs_outcomes_imputed, acs_outcomes_raw=acs_outcomes_raw,
                          demographics=demographics, language_task_scores=language_task_scores,
                          factor_scores_theory=factor_scores_theory,
                          cognitive_overall_score=cognitive_overall_score, transcriptions_google=transcriptions_google,
                          transcriptions_whisper=transcriptions_whisper, transcripts=transcripts_numpy,
                          mean_composite_cognitive_score=mean_composite_cognitive_score,
                          extra_targets=extra_targets, moca_scores=moca_scores,
                          acs_questionnaires=acs_questionnaires, tasks=np.array([self.task] * len(sample_names)))

        if self.task == 'all_individual':
            dataset = self._multiply_dataset_with_individual_spontaneous_speech_tasks(dataset, transcripts, audio_files_df)

        if self.debug:
            n_samples = self.debug if self.debug > 1 else 10
            dataset = dataset.subset_from_indices(list(range(n_samples)))

        cookieTheftTasks = ['cookieTheft', 'cookieTheftDelayedRecall']
        if self.task in cookieTheftTasks:
            transcript_stats = dataset.transcriptions_google.copy()
            transcript_stats['sample_name'] = dataset.sample_names
            transcript_stats = transcript_stats[cookieTheftTasks + ['sample_name']].melt(id_vars='sample_name', var_name='task', value_name="transcript", value_vars=cookieTheftTasks)
            transcript_stats['length'] = transcript_stats['transcript'].apply(lambda x: len(x) if not pd.isna(x) else 0)
            participants_to_exclude = transcript_stats.query("length < 100").sample_name.unique()
            print(f"\n\nExcluding {len(participants_to_exclude)} participants with too short transcript (< 100) on cookieTheft or cookieTheftDelayedRecall tasks: {participants_to_exclude}\n\n")
            dataset = dataset.subset_from_sample_names(list(set(dataset.sample_names) - set(participants_to_exclude)))
            dataset.name = f"{dataset_name} - Removed participants with too short transcript"

        if self.waves is not None:
            wave_filter = np.isin(dataset.waves, self.waves)
            dataset = dataset.subset_from_indices(list(np.where(wave_filter)[0]))
            dataset.name = f"{dataset_name} - Waves={self.waves}"
            assert np.all(np.isin(dataset.waves, self.waves))

        if self.store_loaded_dataset:
            dataset.store_to_disk(os.path.join(self.run_parameters.results_dir, "dataset_after_dataloader"))

        return dataset



class LuhaCombinedDataLoader(LuhaBaseDataLoader):
    """
    Combined LUHA dataloader that loads both 2024 and 2026 datasets and merges them.
    Internally uses Luha2024DataLoader (full split) and Luha2026DataLoader (factor scores compatible with 2024). Only data fields and DataFrame columns common to both datasets are retained in the resulting Dataset.

    Config keys:
        - task: must be in the intersection of 2024/2026 tasks
        - transcript_version: 'google' or 'whisper'
    """

    VALID_TASKS = ['cookieTheft', 'journaling', 'picnicScene', 'pictureDescription', 'pictureNaming', 'all_individual']

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'Luha Combined Dataloader'
        kwargs['dataset_type'] = DatasetType.LUHACombined
        super().__init__(*args, **kwargs)

        try:
            self.only_longitudinal_participants = self.config.config_data.only_longitudinal_participants
        except (AttributeError, KeyError):
            self.only_longitudinal_participants = False

        try:
            self.use_averaged_factor_scores_theory = self.config.config_data.use_averaged_factor_scores_theory
        except (AttributeError, KeyError):
            self.use_averaged_factor_scores_theory = False
        if self.use_averaged_factor_scores_theory:
            assert self.only_longitudinal_participants, "Averaged factor scores are only available for longitudinal participants"

        try:
            self.split_2024 = self.config.config_data.split_2024
        except (AttributeError, KeyError):
            self.split_2024 = 'full'
        assert self.split_2024 in ['full', 'train'], f"Invalid split_2024 '{self.split_2024}'. Must be 'full' or 'train'"

        try:
            self.data_split = str(self.config.config_data.split)
        except (AttributeError, KeyError):
            self.data_split = 'full'
        valid_data_splits = ['full', 'wave1_cohortA']
        assert self.data_split in valid_data_splits, f"Invalid split '{self.data_split}'. Must be in {valid_data_splits}"

        config_dict = self.config.to_dict() if self.config else {}
        config_data = config_dict.get('config_data', {})

        if 'factor_scores_theory_version' in config_data or 'factor_scores_theory_version_2026' in config_data:
            raise ValueError("factor_scores_theory_version is not used anymore, this combined Dataloader always uses the factor scores compatible between 2024 and 2026")

        # 2024 config: load with specified split, cv_splits=1 to allow full dataset
        config_data_2024 = {k: v for k, v in config_data.items()
                            if k not in ('compatible_with_2024', 'split_2024')}
        config_data_2024['split'] = self.split_2024
        self.config_2024 = Config.from_dict({
            'config_data': config_data_2024,
            'config_model': {**config_dict.get('config_model', {}), 'cv_splits': 1},
        })

        # 2026 config: force compatible factor scores standardized to 2024 norms
        config_data_2026 = {k: v for k, v in config_data.items()
                            if k not in ('split', 'split_2024')}
        config_data_2026['compatible_with_2024'] = True
        self.config_2026 = Config.from_dict({
            'config_data': config_data_2026,
        })

        self.loader_2024 = Luha2024DataLoader(
            debug=self.debug, constants=self.CONSTANTS, config=self.config_2024, run_parameters=self.run_parameters
        )
        self.loader_2026 = Luha2026DataLoader(
            debug=self.debug, constants=self.CONSTANTS, config=self.config_2026, run_parameters=self.run_parameters
        )

        print(f"Initializing {self.name} (task={self.task}, transcript_version={self.transcript_version}, split={self.data_split}, split_2024={self.split_2024}, only_longitudinal_participants={self.only_longitudinal_participants}), waves={self.waves}, use_averaged_factor_scores_theory={self.use_averaged_factor_scores_theory}")

    @staticmethod
    def _align_dataframe_columns(df1, df2):
        """Return (df1, df2) trimmed to the intersection of columns, preserving df1's column order."""
        common_cols = [c for c in df1.columns if c in set(df2.columns)]
        return df1[common_cols], df2[common_cols]

    def load_data(self):
        print(f"Loading data using {self.name}")

        dataset_2024 = self.loader_2024.load_data()
        dataset_2026 = self.loader_2026.load_data()

        # rename language columns in 2026 to make them compatible to 2026 data
        dataset_2026.language_task_scores = dataset_2026.language_task_scores.rename(columns={'phonemic_fluency_f_score': 'phonemic_fluency_score'})

        # When using only 2024 training data, exclude 2026 longitudinal participants
        # whose 2024 counterpart is in the 2024 test set (to avoid leakage)
        if self.split_2024 == 'train':
            train_test_split_df = pd.read_csv(self.CONSTANTS.TRAIN_TEST_DATASPLIT, dtype={'study_submission_id': str})
            test_sample_names_2024 = set(train_test_split_df[train_test_split_df.split == 'test']['study_submission_id'].values)
            demographics_2026 = dataset_2026.demographics
            if '2024_sample_name' in demographics_2026.columns:
                longitudinal_in_test = demographics_2026['2024_sample_name'].isin(test_sample_names_2024)
                if longitudinal_in_test.any():
                    keep_sample_names = dataset_2026.sample_names[~longitudinal_in_test.values]
                    n_excluded = longitudinal_in_test.sum()
                    print(f"Excluding {n_excluded} 2026 longitudinal participants whose 2024 counterpart is in the test set")
                    dataset_2026 = dataset_2026.subset_from_sample_names(keep_sample_names)

        # Determine common data variables (sample_names handled separately as a positional arg)
        vars_2024 = set(dataset_2024.data_variables.keys()) - {'sample_names'}
        vars_2026 = set(dataset_2026.data_variables.keys()) - {'sample_names'}
        common_vars = sorted(vars_2024 & vars_2026)

        only_2024 = vars_2024 - vars_2026
        only_2026 = vars_2026 - vars_2024
        if only_2024 or only_2026:
            print(f"  Dropping 2024-only fields: {only_2024}")
            print(f"  Dropping 2026-only fields: {only_2026}")

        combined_sample_names = np.concatenate([dataset_2024.sample_names, dataset_2026.sample_names])

        combined_data = {}
        for var in common_vars:
            val_2024 = dataset_2024.data_variables[var]
            val_2026 = dataset_2026.data_variables[var]

            if isinstance(val_2024, pd.DataFrame) and isinstance(val_2026, pd.DataFrame):
                val_2024, val_2026 = self._align_dataframe_columns(val_2024, val_2026)
                if set(val_2024.columns) != set(dataset_2024.data_variables[var].columns) or \
                   set(val_2026.columns) != set(dataset_2026.data_variables[var].columns):
                    print(f"  {var}: aligned to {len(val_2024.columns)} common columns")

                # Re-attach sample_name so Dataset constructor can validate ordering
                v2024 = val_2024.copy()
                v2024['sample_name'] = dataset_2024.sample_names
                v2026 = val_2026.copy()
                v2026['sample_name'] = dataset_2026.sample_names
                combined_data[var] = pd.concat([v2024, v2026], axis=0).reset_index(drop=True)
            elif isinstance(val_2024, np.ndarray) and isinstance(val_2026, np.ndarray):
                combined_data[var] = np.concatenate([val_2024, val_2026], axis=0)
            else:
                raise TypeError(f"Cannot combine {var}: types {type(val_2024)} and {type(val_2026)}")

        # Year indicator for each sample
        dataset_year = np.array(
            ['2024'] * len(dataset_2024) + ['2026'] * len(dataset_2026)
        )

        # The waves for each participant (i.e. for each sample_name_longitudinal, which waves exist in the data?)
        waves_and_ids = pd.DataFrame({'sample_name': combined_sample_names, 'sample_name_longitudinal': combined_data['sample_names_longitudinal'], 'wave': combined_data['waves']}).drop_duplicates()
        longitudinal_waves_lookup = waves_and_ids.groupby("sample_name_longitudinal").apply(lambda group: pd.Series({'waves': group['wave'].sort_values().to_list()})).reset_index()
        longitudinal_waves_lookup = longitudinal_waves_lookup.merge(waves_and_ids[['sample_name_longitudinal', 'sample_name']], on="sample_name_longitudinal", how="outer").drop(columns=['sample_name_longitudinal'])
        longitudinal_waves = pd.DataFrame({'sample_name': combined_sample_names}).merge(longitudinal_waves_lookup, on="sample_name", how="left")
        assert np.all(longitudinal_waves.sample_name == combined_sample_names), "After merging to get longitudinal waves, sample_name order should not change"
        longitudinal_waves = np.array(longitudinal_waves['waves'])

        dataset_name = f"LUHA Combined 2024+2026 ({self.transcript_version} {self.task})"
        combined_dataset = Dataset(
            name=dataset_name,
            type=DatasetType.LUHACombined,
            sample_names=combined_sample_names,
            config={
                'data_transformers': [],
                'debug': self.debug,
                'data_split': self.data_split,
                'split_2024': self.split_2024,
                'waves': self.waves,
                'only_longitudinal_participants': self.only_longitudinal_participants,
            },
            dataset_year=dataset_year,
            **combined_data,
            longitudinal_waves=longitudinal_waves,
        )

        print(f"Combined dataset: {len(dataset_2024)} (2024) + {len(dataset_2026)} (2026) = {len(combined_dataset)} samples")

        if self.only_longitudinal_participants:
            # if only_longitudinal_participants is True, we keep only participants with at least two waves' data
            waves_and_ids = pd.DataFrame({'sample_name': combined_dataset.sample_names, 'sample_name_longitudinal': combined_dataset.sample_names_longitudinal, 'wave': combined_dataset.waves})
            sample_names_longitudinal_multiple_waves = waves_and_ids.groupby("sample_name_longitudinal")['wave'].nunique().reset_index().query("wave > 1")['sample_name_longitudinal'].to_list()
            longitudinal_participants = waves_and_ids.query("sample_name_longitudinal in @sample_names_longitudinal_multiple_waves")
            sample_names_of_longitudinal_participants = longitudinal_participants['sample_name'].to_list()
            combined_dataset = combined_dataset.subset_from_sample_names(sample_names_of_longitudinal_participants)
            n_participants_per_wave = longitudinal_participants.groupby('wave')['sample_name_longitudinal'].nunique().to_dict()
            n_participants_per_wave_combination = longitudinal_participants.groupby('sample_name_longitudinal').apply(lambda group: pd.Series({'wave_combination': "-".join(group['wave'].sort_values().drop_duplicates().to_list())})).value_counts().to_dict()
            print(f"Only keeping {len(sample_names_of_longitudinal_participants)} samples corresponding to {pd.Series(sample_names_longitudinal_multiple_waves).dropna().nunique()} longitudinal participants ({n_participants_per_wave}) from the following waves: {n_participants_per_wave_combination}")

            if self.use_averaged_factor_scores_theory:
                # factor_scores_theory should be the average of the waves factor scores
                factor_scores_theory = combined_dataset.factor_scores_theory.copy()
                factor_scores_theory['sample_name_longitudinal'] = combined_dataset.sample_names_longitudinal
                factor_scores_theory_new = factor_scores_theory.groupby('sample_name_longitudinal').mean().reset_index()
                factor_scores_theory_new = factor_scores_theory_new.merge(self.waves_and_ids.drop(columns=['wave']), on='sample_name_longitudinal', how='left').drop(columns=['sample_name_longitudinal'])
                assert np.all(factor_scores_theory_new['sample_name_longitudinal'] == combined_dataset.sample_names_longitudinal), "sample_names_longitudinal order should not change. Cannot use _preprocess_pandas_data due to potential task=all_individual"
                assert np.all(factor_scores_theory_new['sample_name'] == combined_dataset.sample_names), "sample_names order should correspond to combined_dataset.sample_names. Cannot use _preprocess_pandas_data due to potential task=all_individual"
                combined_dataset.factor_scores_theory = factor_scores_theory_new

        if self.data_split == 'wave1_cohortA':
            # Keep all wave1 samples + wave2 samples from cohort A (participants present in both wave1 and wave2)
            keep_mask = np.array([
                wave == 'wave1' or (wave == 'wave2' and 'wave1' in lw)
                for wave, lw in zip(combined_dataset.waves, combined_dataset.longitudinal_waves)
            ])
            keep_sample_names = combined_dataset.sample_names[keep_mask]
            n_excluded = (~keep_mask).sum()
            print(f"Split '{self.data_split}': keeping {keep_mask.sum()} samples, excluding {n_excluded} wave2-only / wave3 samples")
            combined_dataset = combined_dataset.subset_from_sample_names(keep_sample_names)
            combined_dataset.name = f"{combined_dataset.name} - Split {self.data_split}"

        return combined_dataset


class DataLoader:
    def __new__(cls, *args, **kwargs):
        dataset_type = kwargs.get('dataset_type', DatasetType.LUHA2024)

        if dataset_type == DatasetType.LUHA2024:
            return Luha2024DataLoader(*args, **kwargs)
        elif dataset_type == DatasetType.LUHA2026:
            return Luha2026DataLoader(*args, **kwargs)
        elif dataset_type == DatasetType.LUHACombined:
            return LuhaCombinedDataLoader(*args, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}")






