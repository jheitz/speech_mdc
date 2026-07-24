import os
import pandas as pd
import shutil
import sys
from datetime import datetime
import logging
import numpy as np


sys.path.insert(0, '..') # to make the import from parent dir work

from config.constants import Constants
from preparation_logic.ACS_outlier_removal_imputation import ACSOutlierRemovalImputation

class ProcessedDataCombiner:
    def __init__(self, constants):
        self.constants = constants
        self.PREPROCESSED_COMBINED_PATH = constants.DATA_PROCESSED_COMBINED_2026
        self.PREPROCESSED_COMBINED_DATA_PATH = os.path.join(self.PREPROCESSED_COMBINED_PATH, 'data')

    def _combine_csv(self, data_paths, csv_path, df_transform=None, dtype=None):
        data = []
        for data_path in data_paths:
            csv_path_full = os.path.join(data_path, csv_path)
            if not os.path.exists(csv_path_full):
                logging.error(f"{csv_path_full} does not exist")
                continue
            df = pd.read_csv(csv_path_full, dtype=dtype)
            if df_transform is not None:
                df = df_transform(df)
            data.append(df)
        concatenated = pd.concat(data, ignore_index=True)
        return concatenated

    def _store_study_csv_files(self, dir_to_store, combined_csv_data: dict[str, pd.DataFrame], submission_id_subset=None):
        os.makedirs(dir_to_store, exist_ok=True)

        for csv_name in combined_csv_data.keys():
            path = os.path.join(dir_to_store, f"{csv_name}.csv")
            data = combined_csv_data[csv_name]
            data['study_submission_id'] = data['study_submission_id'].round(0).astype(int)

            if submission_id_subset is not None:
                data = data[data.study_submission_id.isin(submission_id_subset)]
                if data.shape[0] != len(submission_id_subset):
                    logging.error(
                        f"{len(submission_id_subset)} submission ids given, but only found {data.shape[0]} in {csv_name}, missing ids {[id for id in submission_id_subset if id not in data.study_submission_id.to_list()]}")

            data.sort_values(by="study_submission_id").to_csv(path, index=False)


    def _combine_study_csv_files(self, round_paths, dir_to_store):
        prolific_data = self._combine_csv(round_paths, "prolific_data.csv")
        prolific_data.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "_prolific_data.csv"), index=False)
        study_submissions = self._combine_csv(round_paths, "study_submissions.csv", dtype={'2024_study_submission_id': str})
        study_submissions.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "_study_submissions.csv"), index=False)

        study_submissions[['study_submission_id', '2024_study_submission_id']].dropna().to_csv(os.path.join(dir_to_store, "_longitudinal_participants.csv"), index=False)

        waves_and_ids = study_submissions[['study_submission_id', 'longitudinal_id', 'wave']].copy()
        ids_2024 = pd.read_csv(os.path.join(constants.DATA_RAW_2026, "luha_2024_id_mapping.csv"), dtype=str)[['study_submission_id', 'longitudinal_id']]
        ids_2024['wave'] = 'wave1'
        waves_and_ids = pd.concat([waves_and_ids, ids_2024], ignore_index=True).sort_values(by="study_submission_id", key=lambda x: x.astype(int))
        waves_and_ids.to_csv(os.path.join(dir_to_store, "waves_and_ids.csv"), index=False)

        demographics = self._combine_csv(round_paths, "demographics.csv")
        demographics = demographics.merge(study_submissions[['study_submission_id', '2024_study_submission_id']], on='study_submission_id', how='left')
        demographics.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "demographics.csv"), index=False)

        # ACS scores in three versions: orginal scores / with outlier removal based on norms / with imputation
        # version two is based on the default demographic parameters instead of mouse_type, see kw02/acs_norms for details
        outlier_imputation = ACSOutlierRemovalImputation(constants=self.constants, version_logic=2)
        ## original scores
        acs_outcomes_raw = self._combine_csv(round_paths, "acs_outcomes.csv")
        acs_outcomes_raw.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "acs_outcomes_raw.csv"), index=False)
        ## remove outliers
        acs_outcomes_outliers_removed = outlier_imputation._remove_outliers(acs_outcomes_raw.copy(), demographics)
        acs_outcomes_outliers_removed.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "acs_outcomes_outliers_removed.csv"), index=False)
        ## with imputation of missing values
        acs_outcomes_imputed = outlier_imputation._impute_missing_values(acs_outcomes_outliers_removed, verbose=False)
        acs_outcomes_imputed.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "acs_outcomes_imputed.csv"), index=False)

        health_questionnaire = self._combine_csv(round_paths, "health_questionnaire.csv")
        iadl_questionnaire = self._combine_csv(round_paths, "iadl_questionnaire.csv")
        scd_questionnaire = self._combine_csv(round_paths, "scd_questionnaire.csv")

        name_mapping = {
            'semanticFluency': 'semantic_fluency_score',
            'semanticFluencyVegetables': 'semantic_fluency_vegetables_score',
            'phonemicFluencyF': 'phonemic_fluency_f_score',
            'phonemicFluencyA': 'phonemic_fluency_a_score',
            'phonemicFluencyS': 'phonemic_fluency_s_score',
            'pictureNaming': 'picture_naming_score',
            'storyImmediateRecall': 'story_immediate_recall_score',
            'storyDelayedRecall': 'story_delayed_recall_score'
        }
        language_task_scores = self._combine_csv(round_paths, "language_task_scores.csv", df_transform=lambda df: df.rename(columns=name_mapping))
        moca_scores = self._combine_csv(round_paths, "moca_scores.csv")

        # Note: prolific data and study_submissions should not be needed anymore
        # all necessary information should be in demographics
        # we keep them with a underscore prefix as "internal" data right now, could be removed in the future.
        csv_data = {
            "_prolific_data": prolific_data,
            "_study_submissions": study_submissions,
            "demographics": demographics,
            "acs_outcomes_raw": acs_outcomes_raw,
            "acs_outcomes_outliers_removed": acs_outcomes_outliers_removed,
            "acs_outcomes_imputed": acs_outcomes_imputed,
            "health_questionnaire": health_questionnaire,
            "iadl_questionnaire": iadl_questionnaire,
            "scd_questionnaire": scd_questionnaire,
            "language_task_scores": language_task_scores,
            "moca_scores": moca_scores,
        }

        return csv_data

    def combine_data(self):
        print("Combining preprocessed data:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if os.path.exists(self.PREPROCESSED_COMBINED_PATH):
            print(f"Deleting old process_combined data at {self.PREPROCESSED_COMBINED_PATH}")
            shutil.rmtree(constants.DATA_PROCESSED_COMBINED_2026)
        #os.makedirs(self.PREPROCESSED_COMBINED_DATA_PATH, exist_ok=True)

        round_paths = []
        for round in sorted(os.listdir(constants.DATA_PROCESSED_2026)):
            if os.path.isdir(os.path.join(constants.DATA_PROCESSED_2026, round)):
                source_path = os.path.join(constants.DATA_PROCESSED_2026, round, "data")
                print(f"\nCopying data from round {round} ({source_path}): \n... ", end="")
                assert os.path.exists(source_path), "{} does not exist".format(os.path.join(constants.DATA_PROCESSED_2026, round, "data"))
                round_paths.append(source_path)
                for submission_id in sorted(os.listdir(source_path)):
                    if not submission_id.isdigit():  # other subfolder, such as automatic_test_scoring
                        continue
                    source_dir = os.path.join(source_path, submission_id)
                    if os.path.isdir(source_dir):
                        print(submission_id, end=" ")
                        destination_dir = os.path.join(self.PREPROCESSED_COMBINED_DATA_PATH, submission_id)
                        shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

        print("\nCombining overall CSV data files")
        combined_csv_data = self._combine_study_csv_files(round_paths, dir_to_store=self.PREPROCESSED_COMBINED_DATA_PATH)
        self._store_study_csv_files(self.PREPROCESSED_COMBINED_DATA_PATH, combined_csv_data)


if __name__ == '__main__':
    constants = Constants()
    data_combiner = ProcessedDataCombiner(constants)
    data_combiner.combine_data()
    print("Done!")
