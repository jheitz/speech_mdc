import os, sys


class Constants:
    """
    A class of project-wise CONSTANTS.
    Directory paths can depend on being run locally or not (which is given by the --local runtime parameter)
    This local flag can also be set manually (Constants(True) or Constants(False)), or, in any case,
    the local / remote version can be accessed using Constants().LOCAL.CONSTANT_NAME / Constants().REMOTE.CONSTANT_NAME
    """
    def __init__(self, local=None, recursion=True):
        git_dir_remote = "/home/ubuntu/git/luha-prolific-study"
        git_dir_local = "/Users/jheitz/git/luha-prolific-study"
        git_dir_science_cluster = "/home/jheitz/git/luha-prolific-study"

        if local is not None:
            self.local = local
        else:
            if "--local" in sys.argv:
                self.local = True
            elif os.path.exists(git_dir_local):
                self.local = True
            else:
                self.local = False

        self.science_cluster = os.path.exists(git_dir_science_cluster)

        mounted_methlab_remote = "/home/ubuntu/methlab/Students/Jonathan/"
        if self.local:
            mounted_methlab_remote = "/Volumes/methlab/Students/Jonathan/local"

        # Git branch where code snapshots are committed and pushed to by src/run/run.py
        self.EXPERIMENT_BRANCH = 'experiments'

        # path to directory of project git
        self.GIT_DIR = git_dir_remote
        if self.local:
            self.GIT_DIR = git_dir_local
        elif self.science_cluster:
            self.GIT_DIR = git_dir_science_cluster

        self.CACHE_DIR = os.path.join(self.GIT_DIR, "cache")
        self.CACHE_DIR_CENTRALIZED = os.path.join(mounted_methlab_remote, "luha_cache")  # centralized cache (for all virtual machines)
        if self.local:
            self.CACHE_DIR_CENTRALIZED = os.path.join(self.GIT_DIR, "cache")
        elif self.science_cluster:
            self.CACHE_DIR_CENTRALIZED = "/home/jheitz/luha_cache"
        self.RESOURCES_DIR = os.path.join(self.GIT_DIR, "src/resources")

        self.DATA_RAW = None
        self.DATA_PROCESSED = None
        self.DATA_PROCESSED_COMBINED = os.path.join(mounted_methlab_remote, "data", "prolificStudy", "processed_combined")
        self.DATA_INTERMEDIATES = os.path.join(mounted_methlab_remote, "data_intermediates", "prolificStudy")
        if self.local:
            self.DATA_RAW = os.path.join(self.GIT_DIR, "data", "raw")
            self.DATA_PROCESSED = os.path.join(self.GIT_DIR, "data", "processed")
            self.DATA_PROCESSED_COMBINED = os.path.join(self.GIT_DIR, "data", "processed_combined")
            self.DATA_INTERMEDIATES = os.path.join(self.GIT_DIR, "data", "intermediates")
        elif self.science_cluster:
            self.DATA_PROCESSED_COMBINED = os.path.join("/home/jheitz/data/data/prolificStudy", "processed_combined")
            self.DATA_INTERMEDIATES = os.path.join("/home/jheitz/data/data", "intermediates")

        self.DATA_RAW_2026 = os.path.join(self.GIT_DIR, "data2026", "raw")
        self.DATA_PROCESSED_2026 = os.path.join(self.GIT_DIR, "data2026", "processed")
        self.DATA_PROCESSED_COMBINED_2026 = os.path.join(mounted_methlab_remote, "projects/LanguageHealthyAging/data_round2_2026/data", "processed_combined")
        self.DATA_INTERMEDIATES_2026 = os.path.join(mounted_methlab_remote, "data_intermediates", "prolificStudy2026")
        if self.local:
            self.DATA_PROCESSED_COMBINED_2026 = os.path.join(self.GIT_DIR, "data2026", "processed_combined")
            self.DATA_INTERMEDIATES_2026 = os.path.join(self.GIT_DIR, "data2026", "intermediates")
        if self.science_cluster:
            self.DATA_PROCESSED_COMBINED_2026 = os.path.join("/home/jheitz/data/data_round2_2026/data", "processed_combined")
            self.DATA_INTERMEDIATES_2026 = os.path.join("/home/jheitz/data/data_round2_2026/intermediates")
        self.WAVES_AND_IDS = os.path.join(self.DATA_PROCESSED_COMBINED_2026, 'data', "waves_and_ids.csv")

        self.LUHA_2026_SPONATANEOUS_SPEECH_TASKS = ['cookieTheft', 'cookieTheftImmediateRecall', 'cookieTheftDelayedRecall', 'picnicScene', 'journaling']
        self.LUHA_2026_LANGUAGE_TASKS = ['phonemicFluencaF', 'phonemicFluencaA', 'phonemicFluencaS', 'semanticFluency', 'semanticFluencyVegetables', 'pictureNaming', 'storyImmediateRecall', 'storyDelayedRecall']

        self.TRAIN_TEST_DATASPLIT = os.path.join(self.RESOURCES_DIR, "train_test_split_2025-01-08_17-03.csv")
        self.SPLIT1_SPLIT2_DATASPLIT = os.path.join(self.RESOURCES_DIR, "data_split_2024-07-10_16-11.csv")

        self.CV_FOLD_ASSIGNMENT_CV5 = os.path.join(self.RESOURCES_DIR, "cv_fold_assignment_cv5_2025_09_23.csv")

        self.CV_FOLD_ASSIGNMENT_COMBINED_FULL_CV10 = os.path.join(self.RESOURCES_DIR, "cv_fold_assignment_combined_full_cv10_2026-05-07_10-36.csv")
        self.CV_FOLD_ASSIGNMENT_COMBINED_WAVE1COHORTA_CV10 = os.path.join(self.RESOURCES_DIR, "cv_fold_assignment_combined_wave1_cohortA_cv10_2026-05-12_17-18.csv")
        self.CV_FOLD_ASSIGNMENT_COMBINED_TRAIN_CV5_WITH_VAL = os.path.join(self.RESOURCES_DIR, "cv_fold_assignment_combined_train2024_full2026_cv5_with_val_2026-04-17_08-52.csv")
        self.CV_FOLD_ASSIGNMENT_COMBINED_WAVE1COHORTA_TEST_COHORTB = os.path.join(self.RESOURCES_DIR, "split_assignment_combined_wave1_cohortA_train_rest_test_2026-05-13_11-08.csv")
        self.CV_FOLD_ASSIGNMENT_COMBINED_FULL_CV5_WITH_VAL = os.path.join(self.RESOURCES_DIR, "cv_fold_assignment_combined_full_cv5_2026-05-28_10-29.csv")
        self.CV_FOLD_ASSIGNMENT_COMBINED_WAVE1COHORTA_CV5_WITH_VAL = os.path.join(self.RESOURCES_DIR, "cv_fold_assignment_combined_wave1_cohortA_cv5_2026-05-28_10-29.csv")
        self.CV_FOLD_ASSIGNMENT_COMBINED_WAVE1COHORTA_TEST_COHORTB_WITH_VAL = os.path.join(self.RESOURCES_DIR, "split_assignment_combined_wave1_cohortA_train_rest_test_2026-05-28_10-53.csv")

        self.TRAIN_VAL_TEST_DATASPLIT = os.path.join(self.RESOURCES_DIR, "train_val_test_datasplit_20250924_1604.csv")
        self.TRAIN_VAL_TEST_DATASPLIT_VAL20 = os.path.join(self.RESOURCES_DIR, "train_val_test_datasplit_20251111_1329_val0.2_state42.csv")
        self.TRAIN_VAL_TEST_DATASPLIT_RESAMPLED = os.path.join(self.RESOURCES_DIR, "train_val_test_datasplit_20251111_1331_val0.1_state24.csv")

        self.COGNITIVE_NEGATIVE_OUTLIERS = os.path.join(self.RESOURCES_DIR, "cognitive_negative_outliers_2025-01-21_13-45.csv")

        self.RESULTS_ROOT = os.path.join(mounted_methlab_remote, "results")
        self.RESULTS_ROOT_REMOTE = self.RESULTS_ROOT
        if self.local:
            self.RESULTS_ROOT = os.path.join(self.GIT_DIR, "results")
            # remote, but accessed from local machine
            self.RESULTS_ROOT_REMOTE = "/Volumes/methlab/Students/Jonathan/results"
        elif self.science_cluster:
            self.RESULTS_ROOT = "/home/jheitz/data/results"
            self.RESULTS_ROOT_REMOTE = "/home/jheitz/data/results"

        self.ACS_NORMATIVE_DATA = os.path.join(self.GIT_DIR, "ACS_normative_data", "csv")

        self.ACS_MAIN_OUTCOME_VARIABLES = ['connect_the_dots_I_time_msec',
                                           'connect_the_dots_II_time_msec', 'wordlist_correct_words',
                                           'avg_reaction_speed', 'place_the_beads_total_extra_moves',
                                           'box_tapping_total_correct', 'fill_the_grid_total_time',
                                           'wordlist_delayed_correct_words', 'wordlist_recognition_correct_words',
                                           'digit_sequence_1_correct_series', 'digit_sequence_2_correct_series']

        self.ACS_MAIN_OUTCOME_VARIABLES_EXTENDED = \
            self.ACS_MAIN_OUTCOME_VARIABLES + ['dragskill_time', 'clickskill_time', 'typeskill_time']

        # DementiaBank (contains the ADReSS challenge data as well as the full PITT corpus)
        self.DATA_DEMENTIABANK_ROOT = "/home/ubuntu/methlab/Students/Jonathan/data/dementiabank_extracted"
        if self.local:
            self.DATA_DEMENTIABANK_ROOT = "/Users/jheitz/phd/data/dementiabank_extracted"

        # PITT corpus (DementiaBank), the full corpus, not only the subset used by the ADReSS challenge
        # Audio:       DATA_PITT_ROOT/<Control|Dementia>/<cookie|fluency>/<participant>-<visit>.mp3
        # Transcripts: DATA_PITT_TRANSCRIPTS/<Control|Dementia>/<cookie|fluency|recall|sentence>/<participant>-<visit>.cha
        self.DATA_PITT_ROOT = os.path.join(self.DATA_DEMENTIABANK_ROOT, "Pitt")
        self.DATA_PITT_TRANSCRIPTS = os.path.join(self.DATA_PITT_ROOT, "Transcripts")
        # participant-level metadata (education, sex, age at entry, diagnoses, ...) of the PITT corpus
        self.DATA_PITT_METADATA = os.path.join(self.DATA_PITT_ROOT, "PItt-data.xlsx")

        # composite cognitive scores
        self.FACTOR_SCORES_THEORY_CSV = os.path.join(self.RESOURCES_DIR, f"factor_scores_theory_2025-01-07-1929.csv")
        self.FACTOR_SCORES_THEORY_2026_COMPATIBLE_WITH_2024 = os.path.join(self.RESOURCES_DIR, f"factor_scores_theory_2026_2026-05-06-1036.csv")  # kw19 data


        # create an explicit attribute LOCAL & REMOTE to get the constants of the local or remote environment
        if recursion:
            # make sure this is not done recursively forever (only once -> recursion=False here)
            self.LOCAL = Constants(local=True, recursion=False)
            self.REMOTE = Constants(local=False, recursion=False)






