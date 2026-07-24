import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import ffmpeg
import re
import shutil
import logging
import warnings
import librosa
import sys
import email
from datetime import datetime

from util.helpers import prolific_id_to_longitudinal_id

sys.path.insert(0, '..') # to make the import from parent dir work

from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters
from util.google_speech_transcription import GoogleSpeechTranscriber
from util.whisper_transcription import Whisper_Transcriber
from util.acs_norm_data import OriginalACSNormDataCalculator


class DataPreparator:
    def __init__(self, run_parameters: RunParameters, config: Config, CONSTANTS: Constants, extra_logic_processors):
        self.run_parameters = run_parameters
        self.config = config
        self.CONSTANTS = CONSTANTS

        # raw dir
        self.RAW_DIR = os.path.join(CONSTANTS.DATA_RAW_2026, config.name)
        assert os.path.exists(self.RAW_DIR)

        # results / preprocessed dir
        try:
            self.RESULTS_DIR = os.path.join(run_parameters.results_dir, "data")
        except:
            self.RESULTS_DIR = os.path.join(CONSTANTS.DATA_PROCESSED_2026, config.name, "data")

        # delete old directory if exists, recreate empty
        if os.path.exists(self.RESULTS_DIR):
            shutil.rmtree(self.RESULTS_DIR)
        os.makedirs(self.RESULTS_DIR, exist_ok=False)

        # also delete old status file and log files
        files_to_delete = ['_status.txt', '1_log.txt', '2_datacheck_errorlog.txt',
                           '2_datacheck_log.txt', '2_datacheck_stdout.txt', '3_test_scoring_errorlog.txt',
                           '3_test_scoring_log.txt', '3_test_scoring_stdout.txt', '4_moca_scoring_errorlog.txt',
                           '4_moca_scoring_log.txt', '4_moca_scoring_stdout.txt']
        for filename in files_to_delete:
            file_path = os.path.join(self.RESULTS_DIR, "..", filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # potential extra logic
        self.extra_logic_processors = extra_logic_processors

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if run_parameters.logfile is not None:
            handler = logging.FileHandler(run_parameters.logfile, encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)

    def _run_extra_logic_processor(self, hook):
        for extra_logic_processor in self.extra_logic_processors:
            if hook == 'after_load_raw_data':
                extra_logic_processor.after_load_raw_data(self)
            elif hook == 'after_prepare_data':
                extra_logic_processor.after_prepare_data(self)
            else:
                raise ValueError(f"Invalid extra logic hook {hook}")

    def log_exception(self, message, participant_id=None):
        self.logger.exception(message)
        if participant_id is not None:
            participant_log_dir = os.path.join(self.RESULTS_DIR, str(participant_id), "log")
            os.makedirs(participant_log_dir, exist_ok=True)
            with open(os.path.join(participant_log_dir, "errorlog.txt"), "a") as errorlog:
                errorlog.write(f"Exception: {message}\n")

    def log_error(self, message, participant_id=None):
        self.logger.error(message)
        if participant_id is not None:
            participant_log_dir = os.path.join(self.RESULTS_DIR, str(participant_id), "log")
            os.makedirs(participant_log_dir, exist_ok=True)
            with open(os.path.join(participant_log_dir, "errorlog.txt"), "a") as errorlog:
                errorlog.write(f"Error: {message}\n")

    def log(self, message, participant_id=None, only_in_subdir=False):
        if not only_in_subdir:
            print(message)
        if participant_id is not None:
            participant_log_dir = os.path.join(self.RESULTS_DIR, str(participant_id), "log")
            os.makedirs(participant_log_dir, exist_ok=True)
            with open(os.path.join(participant_log_dir, "log.txt"), "a") as errorlog:
                errorlog.write(f"{message}\n")

    def load_raw_data(self):
        print("Loading raw data...")
        study_id = self.config.study_id

        # Load prolific exported data
        self.prolific = pd.read_csv(os.path.join(self.RAW_DIR, "prolific_export.csv"))
        self.prolific = self.prolific[self.prolific.Status.isin(['AWAITING REVIEW', 'APPROVED'])]
        assert self.prolific.shape[0] > 0, "No data in prolific export with proper status?"
        print(f"... {self.prolific.shape[0]} rows in prolific data for study {study_id}")

        # Study submission table
        assert study_id is not None, "Study id not provided in config file"
        self.study_submissions = pd.read_csv(os.path.join(self.RAW_DIR, "database/study_submissions.csv"))
        self.study_submissions = self.study_submissions.query(f"study_id == '{study_id}'")
        self.study_submissions = self.study_submissions.rename(columns={'id': 'study_submission_id'})
        # keep only those in prolific export
        self.study_submissions = self.study_submissions[self.study_submissions.prolific_id.isin(self.prolific['Participant id'])]
        print(f"... {self.study_submissions.shape[0]} rows in study submissions")
        # to debug below error: self.study_submissions.merge(self.prolific, left_on="prolific_id", right_on="Participant id", how="outer")
        assert self.study_submissions.shape[0] == self.prolific.shape[0], \
            f"Number of study_submissions and prolific rows should be the same, but is {self.study_submissions.shape[0]} vs. {self.prolific.shape[0]}. ProlificID in prolific data but not in study submissions: {set(self.prolific['Participant id']).difference(set(self.study_submissions['prolific_id']))}"

        # audio recording table
        self.audio_recordings = pd.read_csv(os.path.join(self.RAW_DIR, "database/audio_recordings.csv"))
        # keep only those for study_submissions
        self.audio_recordings = self.audio_recordings[self.audio_recordings.study_submission_id.isin(self.study_submissions['study_submission_id'])]
        print(f"... {self.audio_recordings.shape[0]} rows in audio recordings (for {self.audio_recordings.study_submission_id.drop_duplicates().shape[0]} submissions)")
        assert self.audio_recordings.study_submission_id.drop_duplicates().shape[0] == self.study_submissions.shape[0], "There are submissions without audio files?"

        # moca_sections table
        self.moca_sections = pd.read_csv(os.path.join(self.RAW_DIR, "database/moca_sections.csv"))
        # keep only those for study_submissions
        self.moca_sections = self.moca_sections[self.moca_sections.study_submission_id.isin(self.study_submissions['study_submission_id'])]
        print(f"... {self.moca_sections.shape[0]} rows in MoCA sections (for {self.moca_sections.study_submission_id.drop_duplicates().shape[0]} submissions)")

        # ACS tokens
        self.acs_tokens = pd.read_csv(os.path.join(self.RAW_DIR, "database/acs_tokens.csv"))
        assert self.acs_tokens.token.shape[0] == self.acs_tokens.token.drop_duplicates().shape[0], "Duplicate ACS tokens?"
        # keep only those for study_submissions
        self.acs_tokens = self.acs_tokens[self.acs_tokens.id.isin(self.study_submissions['acs_token_id'])]
        print(f"... {self.acs_tokens.shape[0]} rows in ACS tokens")
        if self.acs_tokens.shape[0] != self.study_submissions.shape[0]:
            self.log_error(f"Number of ACS tokens and study_submissions should be the same, but are {self.acs_tokens.shape[0]} != {self.study_submissions.shape[0]}")

        # ACS cognitive scores
        self.cognitive_scores = pd.read_excel(os.path.join(self.RAW_DIR, "ACS_RESULTS_PROCESSED.xlsx"), sheet_name="Export_template")

        self._run_extra_logic_processor('after_load_raw_data')

    def delete_unnecessary_raw_audio_files(self):
        # The raw audio files in files.uzh-speech.ch are a dump from the web server, so they contain all audios ever
        # recorded, including from older runs, unless they were deleted at some point. This increases the disk space
        # for the raw data dirs a lot by introducing redundancies between the runs. To save space, we delete the raw
        # audio files that are not part of the current study round (based on the study_id column).
        # Note that we keep data of any returned submissions, even if the data is not processed.
        print("Deleting unnecessary raw audio files to save space...")
        study_id = self.config.study_id
        study_submissions = pd.read_csv(os.path.join(self.RAW_DIR, "database/study_submissions.csv"))
        study_submissions = study_submissions.query(f"study_id == '{study_id}'")

        audio_recordings = pd.read_csv(os.path.join(self.RAW_DIR, "database/audio_recordings.csv"))
        # keep only those for study_submissions
        audio_recordings = audio_recordings[audio_recordings.study_submission_id.isin(study_submissions['id'])]
        allowed_audio_ids = audio_recordings['id'].drop_duplicates().to_list()

        # iterate through raw audio files and delete those associated to submissions that are not in this round
        audio_file_base_dir = os.path.join(self.RAW_DIR, "files.uzh-speech.ch/p/assets")
        def find_audio_files_to_delete(d, pattern, file_type):
            files_to_delete = []
            for subdir, dirs, files in os.walk(d):
                for file in files:
                    match = pattern.match(file)
                    if match:
                        audio_id = match.group("id")
                        file_path = os.path.join(subdir, file)
                        if int(audio_id) not in allowed_audio_ids:
                            files_to_delete.append({'type': file_type, 'path': file_path})
            return files_to_delete

        chunk_pattern = re.compile(r"^(?P<id>[0-9]+)_[0-9]+_[a-z0-9_]+\.webm$")
        partial_pattern = re.compile(r"^(?P<id>[0-9]+)\.webm\.part$")
        public_pattern = re.compile(r"^(?P<id>[0-9]+)\.webm$")
        audio_files_to_delete = (find_audio_files_to_delete(os.path.join(audio_file_base_dir, 'chunks'), chunk_pattern, "chunk") +
                                 find_audio_files_to_delete(os.path.join(audio_file_base_dir, 'partial'), partial_pattern, "partial") +
                                 find_audio_files_to_delete(os.path.join(audio_file_base_dir, 'public'), public_pattern, "public"))
        audio_files_to_delete_df = pd.DataFrame(audio_files_to_delete)
        audio_files_to_delete_df.to_csv(os.path.join(self.RAW_DIR, "files.uzh-speech.ch", f"deleted_audio_files_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.csv"), index=False)
        if audio_files_to_delete_df.shape[0] == 0:
            print("... No audio files to delete")
            return
        print("... Deleting the following number of files", audio_files_to_delete_df['type'].value_counts().to_dict())
        for _, row in audio_files_to_delete_df.iterrows():
            try:
                os.remove(row['path'])
            except Exception as e:
                self.log_error(f"Error deleting file {row['path']}: {e}")


    def prepare_id_mappings(self):
        # mapping between ACS token, study_submission_id, and prolific_id
        self.id_mapping = self.study_submissions[['study_submission_id', 'prolific_id', 'acs_token_id']]
        self.id_mapping = self.id_mapping.merge(self.acs_tokens[['id', 'token']].rename(columns={
            'id': 'acs_token_id',
            'token': 'acs_token',
        }), on='acs_token_id', how="left")
        self.id_mapping = self.id_mapping.sort_values(by='study_submission_id')
        self.id_mapping = self.id_mapping.drop(columns=['acs_token_id'])

        file_path = os.path.join(self.RESULTS_DIR, "id_mapping.csv")
        print(f"Writing id mapping to {file_path}")
        self.id_mapping.to_csv(file_path, index=False)
        file_path = os.path.join(self.RESULTS_DIR, "id_mapping.txt")
        self.id_mapping.apply(lambda row: " / ".join(row.astype(str)), axis=1).to_csv(file_path, index=False)

    def prepare_prolific_data(self):
        # prepare prolific data: remove unnecessary rows and use the study submission id instead of the prolific_id,
        # as the latter is somewhat private

        print("Preparing Prolific data")

        # check status
        assert np.all(self.prolific['Status'].isin(['AWAITING REVIEW', 'APPROVED']))

        self.prolific = self.prolific.drop(
            columns=["Submission id", "Reviewed at", "Archived at", "Completion code", "Total approvals", "Student status",
                     "Employment status", "Status"]).rename(columns={
            'Participant id': 'prolific_id',
            'Started at': "start_time",
            'Completed at': "end_time",
            'Time taken': "total_time",
            "Age": "age",
            "Sex": "sex",
            "Ethnicity simplified": "ethnicity",
            "Country of birth": "country_of_birth",
            "Country of residence": "country_of_residence",
            "Nationality": "nationality",
            "Language": "language",
        })

        # total time convert to minutes
        self.prolific['total_time'] = self.prolific['total_time'] / 60

        # join study submission id
        n_rows_before = self.prolific.shape[0]
        self.prolific = self.prolific.merge(self.study_submissions[['study_submission_id', 'prolific_id']], on="prolific_id", how="left")
        assert n_rows_before == self.prolific.shape[0], "Duplicate study submission id?"
        assert self.prolific.study_submission_id.isna().sum() == 0, "NA study submission id?"
        self.prolific = self.prolific.drop(columns=['prolific_id'])
        self.prolific = self.prolific[['study_submission_id'] + [c for c in self.prolific.columns if c != "study_submission_id"]]

        prolific_data_path = os.path.join(self.RESULTS_DIR, "prolific_data.csv")
        print(f"Writing preprocessed Prolific data to {prolific_data_path}")
        self.prolific.sort_values(by='study_submission_id').to_csv(prolific_data_path, index=False)

        for _, row in self.prolific.iterrows():
            self.log(f"\nProlific data: {row}\n", row.study_submission_id, only_in_subdir=True)

        return self.prolific

    def extract_acs_feedback(self):
        """
        Extract ACS feedback messages from email files in the 'acs feedback' folder.
        Returns a DataFrame with acs_token and acs_feedback columns.
        """
        print("Extracting ACS feedback from email files...")
        
        feedback_folder = os.path.join(self.RAW_DIR, "acs feedback")
        if not os.path.exists(feedback_folder):
            print(f"... ACS feedback folder not found at {feedback_folder}, skipping feedback extraction")
            return pd.DataFrame(columns=['acs_token', 'acs_feedback'])
        
        feedback_data = []
        email_files = [f for f in os.listdir(feedback_folder) if f.endswith('.eml')]
        
        if len(email_files) == 0:
            print(f"... No email files found in {feedback_folder}")
            return pd.DataFrame(columns=['acs_token', 'acs_feedback'])
        
        print(f"... Found {len(email_files)} email files")
        
        for email_file in email_files:
            email_path = os.path.join(feedback_folder, email_file)
            try:
                with open(email_path, 'rb') as f:
                    msg = email.message_from_bytes(f.read())
                
                # Get the text/plain part
                feedback_text = None
                token = None
                
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == 'text/plain':
                            payload = part.get_payload(decode=True)
                            if payload:
                                feedback_text = payload.decode('utf-8', errors='ignore')
                                break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        feedback_text = payload.decode('utf-8', errors='ignore')
                
                if feedback_text:
                    # Extract token: pattern is "token: <token>" at the end
                    token_match = re.search(r'token:\s*(\S+)', feedback_text, re.IGNORECASE)
                    if token_match:
                        token = token_match.group(1).strip()
                    
                    # Extract feedback message: try to find boundary at browser user agent, fallback to URL
                    # First try to extract up to a browser user agent pattern (Mozilla, Chrome, Safari, etc.)
                    user_agent_pattern = r'(Mozilla|Chrome|Safari|Edge|Firefox|Opera|Version|AppleWebKit)/\d'
                    feedback_match = re.search(r'BERICHT VANAF DE WEBSITE:\s*ACS-LUHA(.*?)' + user_agent_pattern, feedback_text, re.DOTALL | re.IGNORECASE)
                    
                    # If no browser pattern found, use URL as boundary
                    if not feedback_match:
                        feedback_match = re.search(r'BERICHT VANAF DE WEBSITE:\s*ACS-LUHA(.*?)https://acs-luha\.neurotask\.com/tests', feedback_text, re.DOTALL | re.IGNORECASE)
                    
                    if feedback_match:
                        feedback_message = feedback_match.group(1).strip()
                        
                        # Clean up any extra whitespace/newlines
                        feedback_message = re.sub(r'\s+', ' ', feedback_message).strip()
                        
                        if token and feedback_message:
                            feedback_data.append({
                                'acs_token': token,
                                'acs_feedback': feedback_message
                            })
                            print(f"... Extracted feedback for token {token}")
                        elif token:
                            self.log_error(f"Found token {token} but could not extract feedback message from {email_file}")
                        elif feedback_message:
                            self.log_error(f"Found feedback message but could not extract token from {email_file}")
                    else:
                        self.log_error(f"Could not parse feedback format in {email_file}")
                else:
                    self.log_error(f"Could not extract text content from {email_file}")
                    
            except Exception as e:
                self.log_error(f"Error processing email file {email_file}: {e}")
        
        feedback_df = pd.DataFrame(feedback_data)
        
        # Check for duplicate tokens
        if not feedback_df.empty:
            duplicates = feedback_df[feedback_df.duplicated(subset=['acs_token'], keep=False)]
            if not duplicates.empty:
                self.log_error(f"Found duplicate tokens in feedback emails: {duplicates['acs_token'].unique().tolist()}")
                # Keep the first occurrence of each token
                feedback_df = feedback_df.drop_duplicates(subset=['acs_token'], keep='first')
        
        print(f"... Extracted {len(feedback_df)} feedback messages")
        return feedback_df

    def prepare_study_submission_data(self):
        print("Preparing study submissions")

        # add acs_token
        self.study_submissions = self.study_submissions.merge(self.acs_tokens[['id', 'token']].rename(columns={
            'id': 'acs_token_id',
            'token': 'acs_token',
        }), on="acs_token_id", how="left")
        
        # Extract and merge ACS feedback
        acs_feedback_df = self.extract_acs_feedback()
        if not acs_feedback_df.empty:
            # Merge feedback by acs_token
            self.study_submissions = self.study_submissions.merge(
                acs_feedback_df[['acs_token', 'acs_feedback']],
                on='acs_token',
                how='left'
            )
            n_with_feedback = self.study_submissions['acs_feedback'].notna().sum()
            print(f"... {n_with_feedback} submissions have ACS feedback")
        else:
            # Add empty column if no feedback found
            self.study_submissions['acs_feedback'] = None

        print(f"... Calculating laterality index from handedness questionnaire")
        # calculate laterality index from handedness questionnaire based on
        # - https://www.brainmapping.org/shared/Edinburgh.php# /
        # - Oldfield, R.C. "The assessment and analysis of handedness: the Edinburgh inventory." Neuropsychologia. 9(1):97-113. 1971.
        def calc_points(question, id):
            # calculate the points counting towards right: 2 if answer=right, otherHand=False, 1 if answer=right, otherHand=True, 1 if answer=no_preference
            # analogous for left: 2 if answer=left, otherHand=False, 1 if answer=left, otherHand=True, 1 if answer=no_preference
            answer = question['answer']
            other_hand = question['otherHand']
            if answer == "no_preference":
                right, left = 1, 1
            elif answer == "right":
                right = 1 if other_hand else 2
                left = 0
            elif answer == "left":
                left = 1 if other_hand else 2
                right = 0
            else:
                self.log_error(f"Invalid answer for handedness questionaire (submission {id}): {question}", id)
                right, left = None, None
            return {'right': right, 'left': left}

        def handedness(row):
            handedness_json_text = row.handedness
            d = json.loads(handedness_json_text)
            points = pd.DataFrame([calc_points(question, row.study_submission_id) for question in d])
            # display(points)
            if np.sum(points.right) + np.sum(points.left) == 0:
                return None
            index = 100 * (np.sum(points.right) - np.sum(points.left)) / (np.sum(points.right) + np.sum(points.left))
            # print(index)
            return index

        self.study_submissions['laterality_index'] = self.study_submissions.apply(handedness, axis=1)

        # Adding 2024 study submission ID for re-tested individuals (making this a longitudinal dataset)
        # Persistent ID: Prolific_id
        print("... Adding 2024 study submission ID for re-tested individuals")
        id_mapping_2024 = pd.read_csv(os.path.join(self.RAW_DIR, "..", "luha_2024_id_mapping.csv")).rename(columns={
            'study_submission_id': '2024_study_submission_id',
        })[['prolific_id', '2024_study_submission_id']]
        self.study_submissions = self.study_submissions.merge(id_mapping_2024, on='prolific_id', how='left')

        # create longitudinal_id, our own longitudinal ID as a hash of the prolific_id
        self.study_submissions['longitudinal_id'] = self.study_submissions['prolific_id'].apply(lambda id: prolific_id_to_longitudinal_id(id, self.CONSTANTS.GIT_DIR))
        self.study_submissions['wave'] = self.config.wave

        print("... Removing prolific_id")
        # remove prolific_id (as this is somewhat private and should not be shared - we use our own database id for this dataset)
        self.study_submissions = self.study_submissions.drop(columns=['prolific_id'])
        study_submissions_path = os.path.join(self.RESULTS_DIR, "study_submissions.csv")
        print(f"Saving preprocessed study submissions to {study_submissions_path}")
        self.study_submissions.to_csv(study_submissions_path, index=False)

        for _, row in self.study_submissions.iterrows():
            self.log(f"\nStudy submission data: {row}\n", row.study_submission_id, only_in_subdir=True)
            self.log(f"\nFeedback: {row.feedback}", row.study_submission_id, only_in_subdir=True)


        return self.study_submissions

    def create_feedback_file(self):
        """
        Create a feedback.txt file with both webapp feedback and ACS feedback for each submission.
        """
        print("Creating feedback.txt file...")
        
        feedback_file_path = os.path.join(self.RESULTS_DIR, "feedback.txt")
        
        # Get submissions that have either feedback or acs_feedback
        feedback_cols = ['study_submission_id', 'feedback', 'acs_feedback']
        if not all(col in self.study_submissions.columns for col in feedback_cols):
            self.log_error("Missing feedback columns in study_submissions")
            return
        
        feedback_data = self.study_submissions[feedback_cols].copy().sort_values(by='study_submission_id')
                
        with open(feedback_file_path, 'w', encoding='utf-8') as f:
            for _, row in feedback_data.iterrows():
                submission_id = row['study_submission_id']
                webapp_feedback = row['feedback'] if pd.notna(row['feedback']) else ''
                acs_feedback = row['acs_feedback'] if pd.notna(row['acs_feedback']) else ''
                
                # Only write if there's at least one feedback
                if webapp_feedback or acs_feedback:
                    f.write('-' * 15 + '\n')
                    f.write(f'STUDY_SUBMISSION_ID: {submission_id}\n')
                    f.write('ACS FEEDBACK:\n')
                    f.write(f'{acs_feedback}\n')
                    f.write('WEBAPP FEEDBACK:\n')
                    f.write(f'{webapp_feedback}\n')
                    f.write('\n')
        
        print(f"Feedback file created at {feedback_file_path}")

    def check_demographics_consistency(self):
        # check whether the demographics (age & sex/gender) in prolific data match those collected in our study
        study_submission_data = self.study_submissions[['study_submission_id', 'age', 'gender']].rename(columns={
            'age': 'age_study',
            'gender': 'gender_study'
        })
        prolific_data = self.prolific[['study_submission_id', 'age', 'sex']].rename(columns={
            'age': 'age_prolific',
            'sex': 'sex_prolific'
        }).replace({
            'Female': 'f',
            'Male': 'm'
        })
        merged = study_submission_data.merge(prolific_data, on='study_submission_id', how='inner')
        merged['age_diff'] = merged['age_study'].astype(float) - merged['age_prolific'].astype(float)
        for idx, row in merged.query("age_diff != 0").iterrows():
            self.log_exception(f"Participants {row.study_submission_id} has conflicting age information: {row.age_prolific} (Prolific) vs. {row.age_study} (Study Submission)", row.study_submission_id)

        for idx, row in merged.query("sex_prolific != gender_study").iterrows():
            self.log_exception(f"Participants {row.study_submission_id} has conflicting gender information: {row.sex_prolific} (Prolific) vs. {row.gender_study} (Study Submission)", row.study_submission_id)

        print("ok")

    def prepare_demographics(self):
        # Prepare a unified dataframe of demographic information, from Prolific and study_submissions
        prolific_cols = ['study_submission_id', 'sex', 'ethnicity', 'country_of_birth', 'country_of_residence',
                         'nationality', 'age']
        self.demographics = self.prolific[prolific_cols].rename(columns={'age': 'age_prolific', 'sex': 'sex_prolific'})

        study_submission_cols = ['study_submission_id', 'age', 'gender', 'education', 'language', 'country',
                                 'socioeconomic', 'laterality_index', 'consent_data_further_use']
        self.demographics = self.demographics.merge(
            self.study_submissions[study_submission_cols], on='study_submission_id', how='outer'
        )

        # We have Sex information from Prolific, and gender from study_submissions. The latter is sometimes
        # prefer-not-to-say or other, both of which are problematic for norm data (which is calculated per gender)
        # Here, we define the "unified-gender" as the gender information from the study submission (if m / f),
        # or Prolific, if not available.
        def create_unified_gender(row):
            if row['gender'] in ['f', 'm']:
                return 'f' if row['gender'] == 'f' else 'm' if row['gender'] == 'm' else None
            elif row['sex_prolific'] in ['Male', 'Female']:
                return 'f' if row['sex_prolific'] == 'Female' else 'm' if row['sex_prolific'] == 'Male' else None
            else:
                self.logger.error(f"Invalid gender information: study_submission: {row['gender']}, Prolific data: {row['sex_prolific']}")
                return row['gender']
        self.demographics['gender_unified'] = self.demographics.apply(create_unified_gender, axis=1)

        # low vs. high education, based on logic as defined in ACS:
        #    -> "Education" (coded as 0 for Verhage 1-5/ISCED 0-4, 1 for Verhage 6-7/ISCED 5-8)
        # see https://ec.europa.eu/eurostat/statistics-explained/index.php?title=International_Standard_Classification_of_Education_(ISCED)#ISCED_1997_.28fields.29_and_ISCED-F_2013
        self.demographics['education_binary'] = self.demographics.loc[:, 'education'].apply(
            lambda x: 'low-education' if x in ['less_than_highschool', 'high_school', 'vocational'] else 'high-education' if x in ['bachelor', 'master', 'phd'] else x)

        self.demographics.loc[:, 'age'] = self.demographics.loc[:, 'age'].round(0).astype(int)

        demographics_data_path = os.path.join(self.RESULTS_DIR, "demographics.csv")
        print(f"Writing preprocessed demographics data to {demographics_data_path}")
        self.demographics.sort_values(by='study_submission_id').to_csv(demographics_data_path, index=False)

        return self.demographics

    def descriptive_statistics(self):
        # write descriptive statistics
        print("Calculating descriptive statistics")

        # Load questionnaire data if available
        study_submissions_with_questionnaires = self.study_submissions.copy()
        
        # Load SCD questionnaire data
        scd_path = os.path.join(self.RESULTS_DIR, "scd_questionnaire.csv")
        if os.path.exists(scd_path):
            scd_df = pd.read_csv(scd_path)
            study_submissions_with_questionnaires = study_submissions_with_questionnaires.merge(
                scd_df[['study_submission_id', 'total_score', 'total_abc']].rename(columns={'total_score': 'scd_total_score', 'total_abc': 'scd_total_abc'}),
                on='study_submission_id', 
                how='left'
            )
        
        # Load IADL questionnaire data
        iadl_path = os.path.join(self.RESULTS_DIR, "iadl_questionnaire.csv")
        if os.path.exists(iadl_path):
            iadl_df = pd.read_csv(iadl_path)
            study_submissions_with_questionnaires = study_submissions_with_questionnaires.merge(
                iadl_df[['study_submission_id', 'total_score']].rename(columns={'total_score': 'iadl_total_score'}), 
                on='study_submission_id', 
                how='left'
            )
        
        # Load health questionnaire data
        health_path = os.path.join(self.RESULTS_DIR, "health_questionnaire.csv")
        if os.path.exists(health_path):
            health_df = pd.read_csv(health_path)
            study_submissions_with_questionnaires = study_submissions_with_questionnaires.merge(
                health_df[['study_submission_id', 'n_issues']].rename(columns={'n_issues': 'health_issues'}),
                on='study_submission_id', 
                how='left'
            )

        # study submissions
        categorical = ['education', 'gender', 'country', 'language', 'consent_data_further_use']
        continuous = ['age', 'socioeconomic', 'laterality_index']
        
        # Add questionnaire total scores if available
        if 'scd_total_score' in study_submissions_with_questionnaires.columns:
            continuous.append('scd_total_score')  # SCD total_score
        if 'total_abc' in study_submissions_with_questionnaires.columns:
            continuous.append('total_abc')  # SCD total_abc
        if 'iadl_total_score' in study_submissions_with_questionnaires.columns:
            continuous.append('iadl_total_score')  # IADL total_score
        if 'health_issues' in study_submissions_with_questionnaires.columns:
            continuous.append('health_issues')  # Health questionnaire total issues
        
        variables = categorical + continuous

        nrows = int(np.floor(np.sqrt(len(variables))))
        ncols = int(np.ceil(len(variables) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
        axes = [ax for row in axes for ax in row]

        for i in range(len(variables)):
            var = variables[i]
            data = study_submissions_with_questionnaires[var]

            if var in continuous:
                # Filter out NaN values for plotting
                data_clean = data.dropna()
                if len(data_clean) > 0:
                    axes[i].boxplot(data_clean)
                    x = np.random.uniform(0.8, 1.2, size=len(data_clean))
                    axes[i].plot(x, data_clean, ".")
            else:
                categories = pd.Series(data).value_counts().to_dict()
                positions = range(len(categories.keys()))
                axes[i].bar(positions, categories.values())
                axes[i].set_xticks(positions, categories.keys(), rotation=40, ha="right")

            title_parts = var.split("_")
            cumulative_letter_count = np.cumsum([len(part) for part in title_parts])
            diff_to_mean = [np.abs(cumcount - cumulative_letter_count[-1] / 2) for cumcount in cumulative_letter_count]
            argmin = np.argmin(diff_to_mean)
            title_split = "_".join(title_parts[:argmin + 1]) + "\n" + "_".join(title_parts[argmin + 1:])

            axes[i].set_title(title_split, size=10)
            # axes[i].set_ylabel(outcomes[i], size=10)

        plt.suptitle("Study submission data")
        plt.tight_layout()
        plt.savefig(os.path.join(self.RESULTS_DIR, "study_submission_descriptive_stats.png"))
        plt.close()

        study_submission_stats = []
        for i in range(len(variables)):
            var = variables[i]
            data = study_submissions_with_questionnaires[var]

            if var in continuous:
                # Filter out NaN values for statistics
                data_clean = data.dropna()
                if len(data_clean) > 0:
                    study_submission_stats.append({'variable': var, 'distribution': f"{np.mean(data_clean):.2f} +- {np.std(data_clean):.2f}",
                                          "min": f"{np.min(data_clean):.2f}", "max": f"{np.max(data_clean):.2f}"})
            else:
                categories = pd.Series(data).value_counts().to_dict()
                for c in categories.keys():
                    study_submission_stats.append({'variable': f"{var}: {c}",
                                          'distribution': f"{categories[c]} ({int(categories[c] / len(data) * 100)}%)"})

        study_submission_stats_df = pd.DataFrame(study_submission_stats)
        study_submission_stats_df.to_csv(os.path.join(self.RESULTS_DIR, "study_submission_descriptive_stats.csv"), index=False)


        # prolific data
        categorical = ['sex', 'ethnicity', 'country_of_birth', 'country_of_residence', 'nationality', 'language']
        continuous = ['total_time', 'age']
        variables = categorical + continuous

        nrows = int(np.floor(np.sqrt(len(variables))))
        ncols = int(np.ceil(len(variables) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
        axes = [ax for row in axes for ax in row]

        for i in range(len(variables)):
            var = variables[i]
            data = self.prolific[var]

            if var in continuous:
                axes[i].boxplot(data.astype(float))
                x = np.random.uniform(0.8, 1.2, size=len(data))
                axes[i].plot(x, data.astype(float), ".")
            else:
                categories = pd.Series(data).value_counts().to_dict()
                positions = range(len(categories.keys()))
                axes[i].bar(positions, categories.values())
                axes[i].set_xticks(positions, categories.keys(), rotation=40, ha="right")

            title_parts = var.split("_")
            cumulative_letter_count = np.cumsum([len(part) for part in title_parts])
            diff_to_mean = [np.abs(cumcount - cumulative_letter_count[-1] / 2) for cumcount in cumulative_letter_count]
            argmin = np.argmin(diff_to_mean)
            title_split = "_".join(title_parts[:argmin + 1]) + "\n" + "_".join(title_parts[argmin + 1:])

            axes[i].set_title(title_split, size=10)
            # axes[i].set_ylabel(outcomes[i], size=10)

        plt.suptitle("Prolific information")
        plt.tight_layout()
        plt.savefig(os.path.join(self.RESULTS_DIR, "prolific_descriptive_stats.png"))
        plt.close()

        prolific_stats = []
        for i in range(len(variables)):
            var = variables[i]
            data = self.prolific[var]

            if var in continuous:
                data = data.astype(float)
                prolific_stats.append({'variable': var, 'distribution': f"{np.mean(data):.2f} +- {np.std(data):.2f}",
                                      "min": f"{np.min(data):.2f}", "max": f"{np.max(data):.2f}"})
            else:
                categories = pd.Series(data).value_counts().to_dict()
                for c in categories.keys():
                    prolific_stats.append({'variable': f"{var}: {c}",
                                          'distribution': f"{categories[c]} ({int(categories[c] / len(data) * 100)}%)"})

        prolific_stats_df = pd.DataFrame(prolific_stats)
        prolific_stats_df.to_csv(os.path.join(self.RESULTS_DIR, "prolific_descriptive_stats.csv"), index=False)


    def _convert_webm_to_wav(self, input_file, output_file, study_submission_id):
        try:
            # Converting the .webm file to .wav format with overwrite option (-y flag)
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            # Get the metadata of the output file to determine duration
            probe = ffmpeg.probe(output_file)
            duration = float(probe['streams'][0]['duration'])

            self.log(f"Conversion complete. The file '{output_file}' has been written (source: {input_file})", study_submission_id)
            return duration
        except Exception as e:
            self.log_error(f"An error occurred converting webm to wav (study_submission {study_submission_id}, input_file {input_file}, output_file {output_file}):\n{e}", study_submission_id)
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file)
            self.log_error(ffmpeg.run(stream, overwrite_output=True, quiet=False))
            return 0


    @property
    def ACS_outcomes_renaming(self):
        return {
            "O_token": "acs_token",

            # main outcomes
            "O_11_time_test_msec": "connect_the_dots_I_time_msec",
            "O_13_time_test_msec": "connect_the_dots_II_time_msec",
            "O_14_correct_t": "wordlist_correct_words",
            "O_17_RT_average_A": "avg_reaction_speed",
            "O_19_extramoves_t": "place_the_beads_total_extra_moves",
            "O_21_correct_t": "box_tapping_total_correct",
            "O_23_time_test_msec": "fill_the_grid_total_time",
            "O_25_correct_t": "wordlist_delayed_correct_words",
            "O_26_correctPandN": "wordlist_recognition_correct_words",
            "O_28_correct_t": "digit_sequence_1_correct_series",
            "O_30_correct_t": "digit_sequence_2_correct_series",
            "O_32_ANX": "questionnaire_hads_anxiety",
            "O_32_DEPR": "questionnaire_hads_depression",
            "O_33_alg_vermoeidh": "questionnaire_mfi_general_fatigue",
            "O_33_lich_vermoeidh": "questionnaire_mfi_physical_fatigue",
            "O_33_red_activit": "questionnaire_mfi_reduced_activity",
            "O_33_red_motivatie": "questionnaire_mfi_reduced_motivation",
            "O_33_ment_vermoeidh": "questionnaire_mfi_mental_fatigue",
            "O_34_computer_experience": "computer_experience",
            "O_34_type": 'mouse_type',

            # extra outcomes
            "connect_the_dots_difference": "connect_the_dots_difference",
            "connect_the_dots_fraction": "connect_the_dots_fraction",
            "digit_sequence_difference": "digit_sequence_difference",
            "digit_sequence_fraction": "digit_sequence_fraction",
            'O_14_learning_curve1': 'wordlist_correct_trial1',
            'O_14_learning_curve2': 'wordlist_correct_trial2',
            'O_14_learning_curve3': 'wordlist_correct_trial3',
            'O_14_learning_curve4': 'wordlist_correct_trial4',
            'O_14_learning_curve5': 'wordlist_correct_trial5',
            'wordlist_learning': 'wordlist_learning',
            'O_19_extramoves_per_trial': "place_the_beads_extramoves_per_trial",
            'O_19_n_perfect_solutions': "place_the_beads_n_perfect_solutions",
            'O_19_n_solved_in_max_moves': "place_the_beads_n_solved_in_max_moves",
            'O_21_span': 'box_tapping_span_2x',
            'O_21_lcs':  'box_tapping_span_1x',
            'O_28_span': 'digit_sequence_1_span_2x',
            'O_28_lcs': 'digit_sequence_1_span_1x',
            'O_30_span': 'digit_sequence_2_span_2x',
            'O_30_lcs': 'digit_sequence_2_span_1x',
            'O_7_LD': 'typeskill_levenstein_distance',
            'O_7_time_test': 'typeskill_time',
            'O_8_time_test': 'clickskill_time',
            'O_8_finished': 'clickskill_finished',
            'O_8_errors_t': 'clickskill_errors',
            'O_8_missed_t': 'clickskill_misses',
            'O_9_time_test': 'dragskill_time',
            'O_9_optimal_t': 'dragskill_optimal_moves',
            'O_9_drops_t': 'dragskill_drops',
            'O_9_finished': 'dragskill_finished',

            'O_14_time_t': 'wordlist_total_time',
            'O_17_SD_A': 'reaction_speed_std',
            'O_17_time_t': 'reaction_speed_time',
            'O_19_tot_time_10': 'place_the_beads_total_time',
            'O_21_time_t': 'box_tapping_total_time',
            'O_25_time_t': 'wordlist_delayed_total_time',
            'O_26_time_t': 'wordlist_recognition_total_time',
            'O_28_time_t': 'digit_sequence_1_total_time',
            'O_30_time_t': 'digit_sequence_2_total_time',
        }

    def prepare_cognitive_scores(self):
        print("Preparing cognitive scores")

        # keep only cognitive scores for this study, by filtering for relevant tokens
        self.cognitive_scores = self.cognitive_scores[self.cognitive_scores['O_token'].isin(self.study_submissions.acs_token)]
        print(f"... {self.cognitive_scores.shape[0]} rows in cognitive scores")

        if self.cognitive_scores.shape[0] < self.study_submissions.shape[0]:
            self.logger.error("Missing cognitive (ACS) scores for ")

        missing_cognitive_scores = self.study_submissions[~self.study_submissions.acs_token.isin(self.cognitive_scores['O_token'])].copy()
        missing_cognitive_scores = missing_cognitive_scores.merge(self.id_mapping, on=["study_submission_id", "acs_token"], how="left")
        if missing_cognitive_scores.shape[0] > 0:
            self.log_error(f"{missing_cognitive_scores.shape[0]} participants with missing ACS cognitive scores:")
            for _, (study_submission_id, prolific_id, acs_token) in missing_cognitive_scores[['study_submission_id', 'prolific_id', 'acs_token']].iterrows():
                self.log_error(f"Missing ACS cognitive scores for submission {study_submission_id} / {prolific_id} / {acs_token}", study_submission_id)


        # Calculate some additional computed outcomes from raw scores
        def calculate_additional_outcomes(df):
            # O_19_extramoves_t = place the beads - nr moves - nr optimal moves; sum over trials
            # Total number of additional steps required. You must calculate this outcome measure yourself and add it to the data: Calculate for each trial the difference between the number of steps taken and the lowest possible (optimal) number of steps. Take the sum of all trials (O_19_moves_1 - O_19_opt_moves_1 + O_19_moves_2 - O_19_opt_moves_2 + ... + O_19_moves_10 - O_19_opt_moves_10). Note: if not solved within the time (O_19_solved_1 = 0), or needed >20 steps, calculate a maximum score of 20 for the number of steps taken (O_19_moves_1=20).

            # CANTAB also calculates the following outcomes:
            # - % Perfect Solutions
            # - Average Excess moves per trial
            # - % Completed in Maximum moves

            # go through 10 substeps of O_19 (place the beads)
            extra_moves_columns = []
            solved_in_max_moves_columns = []
            for i in range(1, 11):
                is_solved = df[f'O_19_solved_{i}']
                moves = df[f'O_19_moves_{i}']
                optimal_moves = df[f'O_19_opt_moves_{i}']

                new_col = f"0_19_solved_in_max_moves_{i}"
                df[new_col] = np.where((is_solved == 0) | (moves > 20), 0, 1)
                solved_in_max_moves_columns.append(new_col)

                moves_processed = np.where((is_solved == 0) | (moves > 20), 20, moves)
                extra_moves = moves_processed - optimal_moves

                new_col = f'O_19_extramoves_{i}'
                df[new_col] = extra_moves
                extra_moves_columns.append(f'O_19_extramoves_{i}')

            print("     Relevant columns to calculate O_19_extramoves_t (place the beads)", extra_moves_columns)
            df["O_19_extramoves_t"] = df[extra_moves_columns].apply(sum, axis=1)
            df["O_19_extramoves_per_trial"] = df[extra_moves_columns].apply(sum, axis=1) / len(extra_moves_columns)
            df["O_19_n_perfect_solutions"] = (df[extra_moves_columns] == 0).apply(sum, axis=1)
            df["O_19_n_solved_in_max_moves"] = df[solved_in_max_moves_columns].apply(sum, axis=1)

            # O_26_correctPandN = wordlist recognition - nr correct target words + nr correct non-target words
            # You must calculate this outcome measure yourself and add it to the data: O_26_correctP_t + O_26_correctN_t
            df["O_26_correctPandN"] = df['O_26_correctP_t'] + df['O_26_correctN_t']

            # Trails B - Trails A difference. → Thought to be  purer measures of the more complex divided attention and alternating sequencing tasks required in part B
            df['connect_the_dots_difference'] = df['O_13_time_test_msec'] - df['O_11_time_test_msec']
            try:
                df['connect_the_dots_fraction'] = df['O_13_time_test_msec'].astype(float) / df['O_11_time_test_msec']
            except:
                pass

            df['digit_sequence_difference'] = df['O_30_correct_t'] - df['O_28_correct_t']
            try:
                df['digit_sequence_fraction'] = df['O_30_correct_t'].astype(float) / df['O_28_correct_t']
            except:
                pass

            df['wordlist_learning'] = df['O_14_learning_curve5'] - df['O_14_learning_curve1']

            return df

        print("... Calculate additional outputs")
        self.cognitive_scores = calculate_additional_outcomes(self.cognitive_scores)
        cognitive_scores_raw = self.cognitive_scores.copy()

        print("... Rename outcome variables")
        self.cognitive_scores = self.cognitive_scores[self.ACS_outcomes_renaming.keys()].rename(columns=self.ACS_outcomes_renaming)

        # replace acs_token by study_submission_id
        self.cognitive_scores = self.cognitive_scores.merge(self.id_mapping[['acs_token', 'study_submission_id']], on="acs_token", how="left")
        self.cognitive_scores = self.cognitive_scores.drop(columns=['acs_token'])
        self.cognitive_scores = self.cognitive_scores[['study_submission_id'] + [c for c in self.cognitive_scores if c != 'study_submission_id']]

        outcomes_filepath = os.path.join(self.RESULTS_DIR, "acs_outcomes.csv")
        print(f"Writing to {outcomes_filepath}")
        self.cognitive_scores.to_csv(outcomes_filepath, index=False)

        incomplete_cognitive_scores = self.cognitive_scores[self.cognitive_scores.isna().any(axis=1)].copy()
        incomplete_cognitive_scores = incomplete_cognitive_scores.merge(self.id_mapping, on="study_submission_id", how="left")
        if incomplete_cognitive_scores.shape[0] > 0:
            self.log_error(f"{incomplete_cognitive_scores.shape[0]} participants with incomplete ACS cognitive scores:")
            for _, (study_submission_id, prolific_id, acs_token) in incomplete_cognitive_scores[['study_submission_id', 'prolific_id', 'acs_token']].iterrows():
                self.log_error(f"Incomplete ACS cognitive scores for submission {study_submission_id} / {prolific_id} / {acs_token}", study_submission_id)

        self.calculate_norm_scores(self.cognitive_scores)

        return self.cognitive_scores

    def calculate_norm_scores(self, cognitive_scores):
        # norm data, original logic provided by ACS team
        acs_norm_data_calculator = OriginalACSNormDataCalculator(CONSTANTS=self.CONSTANTS, log_problems=True)

        print("Calculating normative scores of ACS")
        data = cognitive_scores.merge(self.demographics, on="study_submission_id", how='left')
        ACS_norm_scores = acs_norm_data_calculator.calculate_norm_scores(data)

        outcomes_filepath = os.path.join(self.RESULTS_DIR, "acs_norm_scores.csv")
        print(f"Writing to {outcomes_filepath}")
        ACS_norm_scores.to_csv(outcomes_filepath, index=False)

        self.plot_cognitive_norm_scores(ACS_norm_scores)

    def plot_cognitive_norm_scores(self, ACS_norm_scores):
        # plot the scores, to detect any outliers
        fig, ax = plt.subplots(figsize=(10, 5))

        def mean_z(row):
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                nanmean = np.nanmean(row[[c for c in row.index if '_Z' in c]])
            return nanmean

        ACS_norm_scores['mean_z'] = ACS_norm_scores.apply(mean_z, axis=1)
        ACS_norm_scores['sorting'] = ACS_norm_scores['mean_z'].rank()
        ACS_norm_scores = ACS_norm_scores.sort_values(by='sorting')

        for col in [c for c in ACS_norm_scores.columns if "_Z" in c]:
            width = 0.5
            if 'wordlist' in col or 'digit_sequence' in col:
                width = 2
            ax.plot(ACS_norm_scores.sorting, ACS_norm_scores[col], ".-", label=col, linewidth=width)

        ax.set_xlabel("Study submission id")
        ax.set_ylabel("z score")
        ax.legend(ncol=3, fontsize='small')
        ax.set_xticks(ACS_norm_scores.sorting, ACS_norm_scores.study_submission_id, rotation=50, ha="right")
        ax.set_ylim([-4, 6])
        ax.set_title("ACS normative scores for participants")
        plt.tight_layout()
        filepath = os.path.join(self.RESULTS_DIR, "acs_norm_scores.png")
        print(f"Plotting norm scores to {filepath}")
        plt.savefig(filepath)
        plt.close()

    def _check_audio_chunks(self, submission_id, task, audio_id, duration):
        # check if there are more chunks than expected. This usually means there was something wrong
        # while recording, which should be looked at separately.
        if task == 'check':
            return

        chunks_base_dir = os.path.join(self.RAW_DIR, "files.uzh-speech.ch/p/assets/chunks/")
        if not os.path.exists(chunks_base_dir):
            self.logger.exception("Raw chunk files do not exist. Cannot do checks")
            return
        chunks_for_audio = [file for file in os.listdir(chunks_base_dir) if file.startswith(f"{audio_id}_")]
        n_chunks = len(chunks_for_audio)

        # if all chunks are in the audio, the duration should be around 10 seconds * n_chunks
        if duration * 1.02 < (n_chunks-1) * 10:
            self.log_exception(f"Audio {audio_id} (submission {submission_id}, {task}) has more raw chunks ({n_chunks}) than expected. Expecting between {(n_chunks-1)* 10}s - {n_chunks*10}s, but got {duration} seconds. ", submission_id)

        # if the hash in the chunk name is not always the same, that's also a sign of something wrong
        chunk_hashes = [re.match("[0-9]+_[0-9]+_(.*)\.[a-zA-Z]{2,5}", chunk).group(1) for chunk in chunks_for_audio]
        if len(set(chunk_hashes)) > 1:
            self.log_exception(f"Audio {audio_id} (submission {submission_id}, {task}) has multiple hashes in the chunks ({set(chunk_hashes)}) which is a sign for something going wrong while recording.", submission_id)

    def copy_recorded_data(self):
        print("Preparing data")
        audio_file_base_dir = os.path.join(self.RAW_DIR, "files.uzh-speech.ch/p/assets/public/")
        moca_file_base_dir = os.path.join(self.RAW_DIR, "files.uzh-speech.ch/p/assets/public/moca/")
        for id in self.study_submissions.study_submission_id.sort_values():
            self.log(f"Analyzing submission id {id}")
            data_dir = os.path.join(self.RESULTS_DIR, str(id))
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(os.path.join(data_dir, "moca"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "audio"), exist_ok=True)
            submission = self.study_submissions.set_index("study_submission_id").loc[id]
            submission.reset_index().to_csv(os.path.join(data_dir, "submission.csv"), index=False)

            moca_participants = os.listdir(moca_file_base_dir)
            if str(id) in moca_participants:
                moca_filenames = [filename for filename in os.listdir(os.path.join(moca_file_base_dir, str(id)))]
                for filename in [f for f in moca_filenames if f in ['moca-clock.png', 'moca-cube.png']]:
                    src_file = os.path.join(moca_file_base_dir, str(id), filename)
                    target_file = os.path.join(data_dir, "moca", filename)
                    shutil.copyfile(src_file, target_file)

            audio_recordings_here = self.audio_recordings.query(f"study_submission_id == {id}")
            # assert audio_recordings_here.task.shape[0] == audio_recordings_here.task.drop_duplicates().shape[0]
            nth_task_repetition = audio_recordings_here.groupby("task")['created_at'].rank()
            audio_durations = []
            moca_audio_durations = []
            for (_, audio), nth_repetition in zip(audio_recordings_here.iterrows(), nth_task_repetition):
                task, audio_id = audio.task, audio.id
                if not audio.finished:
                    msg = f"Audio {audio_id} not finished?"
                    self.log_exception(msg, id)
                audio_file = os.path.join(audio_file_base_dir, f"{audio_id}.webm")
                
                # Check if this is a MoCA audio recording
                is_moca_task = task.startswith("moca-")
                if is_moca_task:
                    # Store MoCA audio recordings in the moca subfolder
                    audio_target_file = os.path.join(data_dir, "moca", f"{task}{str(int(nth_repetition)) if nth_repetition > 1 else ''}.wav")
                else:
                    # Store non-MoCA audio recordings in audio subfolder
                    audio_target_file = os.path.join(data_dir, "audio", f"{task}{str(int(nth_repetition)) if nth_repetition > 1 else ''}.wav")
                
                if audio.task != "check" and nth_repetition > 1:
                    all_ids = audio_recordings_here.sort_values(by=['created_at']).query(f"task == '{task}'")[['id', 'size']]
                    all_ids_str = " / ".join([f"{row.id} ({row.size / 1024 / 1024:.1f}MB)" for row in all_ids.itertuples()])
                    self.log_exception(f"There are {nth_repetition} versions of {task} audio for submission {audio.study_submission_id}: {all_ids_str}", id)
                if not os.path.exists(audio_file):
                    msg = f"Audio file {audio_id} (task {audio.task}, submission {audio.study_submission_id}) does not exist"
                    self.log_exception(msg, id)
                    continue
                # convert to wav and copy to processed dir
                duration = self._convert_webm_to_wav(audio_file, audio_target_file, id)
                self._check_audio_chunks(id, task, audio_id, duration)
                
                # Add to appropriate durations list
                duration_record = {'duration': duration, 'task': task}
                if is_moca_task:
                    moca_audio_durations.append(duration_record)
                else:
                    audio_durations.append(duration_record)
            
            # Save non-MoCA audio durations to main directory
            if len(audio_durations) > 0:
                audio_durations_df = pd.DataFrame(audio_durations)
                audio_durations_df.to_csv(os.path.join(data_dir, "audio_durations.csv"), index=False)
            
            # Save MoCA audio durations to moca subfolder
            if len(moca_audio_durations) > 0:
                moca_dir = os.path.join(data_dir, "moca")
                os.makedirs(moca_dir, exist_ok=True)
                moca_audio_durations_df = pd.DataFrame(moca_audio_durations)
                moca_audio_durations_df.to_csv(os.path.join(moca_dir, "audio_durations.csv"), index=False)

            moca_sections_here = self.moca_sections.query(f"study_submission_id == {id}")
            moca_sections_here.to_csv(os.path.join(data_dir, "moca", "moca_sections.csv"), index=False)

        print("done")

    def check_moca_submission(self):
        """
        Check MoCA submission completeness for each participant.
        Verifies:
        1. Availability of audio files for specific MoCA tasks
        2. Availability of rows in moca_sections.csv for specific sections
        3. That no rows in moca_sections.csv have {'skipped': true} in JSON content
        """
        print("Checking MoCA submission completeness...")
        
        expected_audio_tasks = [
            'moca-naming',
            'moca-memory-trial1',
            'moca-memory-trial2',
            'moca-attention-forward',
            'moca-attention-backward',
            'moca-attention-serial7',
            'moca-sentence-repetition-1',
            'moca-sentence-repetition-2',
            'moca-delayed-recall'
        ]
        expected_sections = [
            'trail-making',
            'copy-cube',
            'draw-clock',
            'attention-letters',
            'abstraction',
            'orientation'
        ]
        all_missing_tasks = []
        
        for submission_id in self.study_submissions.study_submission_id.sort_values():
            data_dir = os.path.join(self.RESULTS_DIR, str(submission_id))
            moca_dir = os.path.join(data_dir, "moca")
            
            missing_tasks_for_submission = []
            
            # Check audio files (only if moca directory exists)
            if os.path.exists(moca_dir):
                for task in expected_audio_tasks:
                    audio_file = os.path.join(moca_dir, f"{task}.wav")
                    if not os.path.exists(audio_file):
                        missing_tasks_for_submission.append(f"audio:{task}")
            else:
                # If moca directory doesn't exist, all audio tasks are missing
                for task in expected_audio_tasks:
                    missing_tasks_for_submission.append(f"audio:{task}")
            
            # Check moca_sections.csv
            moca_sections_path = os.path.join(moca_dir, "moca_sections.csv")
            if os.path.exists(moca_sections_path):
                try:
                    moca_sections_df = pd.read_csv(moca_sections_path)
                    
                    # Check for expected sections
                    if 'section' in moca_sections_df.columns:
                        for section in expected_sections:
                            section_rows = moca_sections_df[moca_sections_df['section'] == section]
                            if section_rows.empty:
                                missing_tasks_for_submission.append(f"section:{section}")
                    else:
                        # If section column doesn't exist, all sections are missing
                        for section in expected_sections:
                            missing_tasks_for_submission.append(f"section:{section}")
                    
                    # Check for skipped sections (check all rows for skipped: true in JSON)
                    if 'data' in moca_sections_df.columns:
                        for idx, row in moca_sections_df.iterrows():
                            data_str = row.get('data', '')
                            if pd.notna(data_str):
                                try:
                                    if isinstance(data_str, str):
                                        data = json.loads(data_str)
                                    else:
                                        data = data_str
                                    
                                    # Check if skipped is true
                                    if isinstance(data, dict) and data.get('skipped') is True:
                                        section_name = row.get('section', 'unknown')
                                        missing_tasks_for_submission.append(f"skipped:{section_name}")
                                except (json.JSONDecodeError, TypeError):
                                    # If JSON parsing fails, skip this check
                                    pass
                except Exception as e:
                    self.log_error(f"Error reading moca_sections.csv for submission {submission_id}: {e}", submission_id)
                    missing_tasks_for_submission.append(f"error:could_not_read_moca_sections")
            else:
                # If moca_sections.csv doesn't exist, all sections are missing
                for section in expected_sections:
                    missing_tasks_for_submission.append(f"section:{section}")
            
            if missing_tasks_for_submission:
                all_missing_tasks.append({
                    'submission_id': submission_id,
                    'missing_tasks': missing_tasks_for_submission
                })
        
        # Print error if any tasks are missing
        if all_missing_tasks:
            error_msg = f"\nERROR: MoCA submission check found missing tasks for {len(all_missing_tasks)} submission(s):\n"
            for entry in all_missing_tasks:
                error_msg += f"  Submission {entry['submission_id']}: {', '.join(entry['missing_tasks'])}\n"
            self.log_error(error_msg)
            print(error_msg)
        else:
            print("... All MoCA submissions are complete.")

    def _transcribe_files_in_dir(self, submission_id, base_dir, google_transcriber, whisper_transcriber):
        transcriptions = []
        os.makedirs(os.path.join(base_dir, "ASR", "google_speech"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "ASR", "whisper"), exist_ok=True)
        for file in os.listdir(base_dir):
            if file.endswith(".wav"):
                audio_file_path = os.path.join(base_dir, file)
                audio_duration = librosa.get_duration(path=audio_file_path)

                try:
                    google_raw = google_transcriber.transcribe_file(audio_file_path)['combined_transcript']
                except Exception as e:
                    self.log_error(f"Error transcribing with Google Speech for submission {submission_id}, file {file}: {e}", submission_id)
                    google_raw = ""
                try:
                    whisper_raw = whisper_transcriber.transcribe_file(audio_file_path)['text']
                except Exception as e:
                    self.log_error(f"Error transcribing with Whisper for submission {submission_id}, file {file}: {e}", submission_id)
                    whisper_raw = ""

                google = google_raw.strip().lower()
                whisper = whisper_raw.strip().lower()

                if audio_duration > 10 * 60:
                    self.log_exception(f"ATTENTION: File {file} for submission {submission_id} is longer than 10 minutes!", submission_id)


                task = file.split(".")[0]
                transcription_file_path_google = os.path.join(base_dir, "ASR", "google_speech", task + ".txt")
                transcription_file_path_whisper = os.path.join(base_dir, "ASR", "whisper", task + ".txt")

                with open(transcription_file_path_google, "w") as f:
                    f.write(google)
                with open(transcription_file_path_whisper, "w") as f:
                    f.write(whisper)

                transcriptions.append({
                    'task': task,
                    'submission': submission_id,
                    'text_google': google,
                    'text_whisper': whisper,
                    'audio_duration': audio_duration
                })

        pd.DataFrame(transcriptions).to_csv(os.path.join(base_dir, "ASR", "transcriptions.csv"), index=False)

    def run_speech_recognition(self):
        print("Transcribing audio files")
        whisper_transcriber = Whisper_Transcriber(use_vad=True)

        for id in self.study_submissions.study_submission_id.sort_values():
            data_dir = os.path.join(self.RESULTS_DIR, str(id))
            submission = self.study_submissions.set_index("study_submission_id").loc[id]
            # language code based on submission
            if submission['language'] in ['english_american', 'english_other']:
                language_code = 'en-US'
            elif submission['language'] in ['english_british']:
                language_code = 'en-GB'
            else:
                raise ValueError(f"Invalid submission language {submission['language']}")

            google_transcriber = GoogleSpeechTranscriber(language_code=language_code)

            # Process audio files in audio subfolder
            audio_dir = os.path.join(data_dir, "audio")
            self._transcribe_files_in_dir(id, audio_dir, google_transcriber, whisper_transcriber)

            # Process MoCA audio files in moca subfolder
            moca_dir = os.path.join(data_dir, "moca")
            if os.path.exists(moca_dir):
                self._transcribe_files_in_dir(id, moca_dir, google_transcriber, whisper_transcriber)

        print("Done transcribing audio files.")

    def prepare_questionnaire_data(self):
        """
        Extract and process questionnaire data (SCD, IADL, health) from study_submissions JSON columns.
        Saves each questionnaire as a CSV file with one row per participant.
        """
        print("Preparing questionnaire data...")

        # Process SCD questionnaire
        print("... Processing SCD questionnaire")
        scd_data = []

        for _, row in self.study_submissions.iterrows():
            scd_record = {'study_submission_id': row['study_submission_id']}

            try:
                if pd.notna(row['SCD']):
                    scd_json = json.loads(row['SCD']) if isinstance(row['SCD'], str) else row['SCD']

                    # Extract all keys as columns
                    for key, value in scd_json.items():
                        scd_record[key] = value

                    # Calculate scores
                    # total_score: number of Yes (true) in questions 1-24
                    # Check for both formats: q01-q24 or question1-question24
                    yes_count = 0
                    processed_keys = set()
                    for i in range(1, 25):
                        # Try different key formats
                        for key_format in [f'q{i:02d}', f'q{i}', f'question{i}']:
                            if key_format in scd_json and key_format not in processed_keys:
                                value = scd_json[key_format]
                                if value is True or value == 'yes' or value == 'Yes' or value == 'true' or value == 'True':
                                    yes_count += 1
                                processed_keys.add(key_format)
                                break

                    scd_record['total_score'] = yes_count

                    # total_abc: number of Yes in a, b, c questions
                    # Check for both formats: a/b/c or a_difficulties/b_ask_doctor/c_worried
                    abc_count = 0
                    abc_keys = ['a', 'b', 'c', 'a_difficulties', 'b_ask_doctor', 'c_worried']
                    for key in abc_keys:
                        if key in scd_json:
                            value = scd_json[key]
                            if value is True or value == 'yes' or value == 'Yes' or value == 'true' or value == 'True':
                                abc_count += 1

                    scd_record['total_abc'] = abc_count
                else:
                    scd_record['total_score'] = None
                    scd_record['total_abc'] = None
                    self.log_error(f"SCD data is missing for submission {row['study_submission_id']}", row['study_submission_id'])
            except Exception as e:
                self.log_error(f"Error processing SCD data for submission {row['study_submission_id']}: {e}", row['study_submission_id'])
                scd_record['total_score'] = None
                scd_record['total_abc'] = None

            scd_data.append(scd_record)

        scd_df = pd.DataFrame(scd_data)
        scd_path = os.path.join(self.RESULTS_DIR, "scd_questionnaire.csv")
        print(f"Writing SCD questionnaire data to {scd_path}")
        scd_df.to_csv(scd_path, index=False)
        
        # Process IADL questionnaire
        print("... Processing IADL questionnaire")
        iadl_data = []

        # IADL scoring based on Lawton IADL scale
        # Manual: https://hign.org/sites/default/files/2020-06/Try_This_General_Assessment_23.pdf
        # Each category scores 0 or 1 based on specific thresholds
        # Categories: telephone, shopping, food_preparation, housekeeping, laundry, transportation, medications, finances
        for _, row in self.study_submissions.iterrows():
            iadl_record = {'study_submission_id': row['study_submission_id']}

            try:
                if pd.notna(row['IADL']):
                    iadl_json = json.loads(row['IADL']) if isinstance(row['IADL'], str) else row['IADL']

                    # Extract all keys as columns
                    for key, value in iadl_json.items():
                        iadl_record[key] = value

                    # Calculate IADL total score (0-8)
                    # Each category gets 0 or 1 point based on specific thresholds:
                    # - telephone: 1 if value < 3, 0 otherwise
                    # - shopping: 1 if value = 0, 0 otherwise
                    # - food_preparation: 1 if value = 0, 0 otherwise
                    # - housekeeping: 1 if value < 4, 0 otherwise
                    # - laundry: 1 if value < 2, 0 otherwise
                    # - transportation: 1 if value < 3, 0 otherwise
                    # - medications: 1 if value = 0, 0 otherwise
                    # - finances: 1 if value < 2, 0 otherwise
                    total_score = 0
                    
                    # Define scoring thresholds for each category
                    scoring_rules = {
                        'telephone': lambda v: 1 if (isinstance(v, (int, float)) and v < 3) else 0,
                        'shopping': lambda v: 1 if (isinstance(v, (int, float)) and v == 0) else 0,
                        'food_preparation': lambda v: 1 if (isinstance(v, (int, float)) and v == 0) else 0,
                        'housekeeping': lambda v: 1 if (isinstance(v, (int, float)) and v < 4) else 0,
                        'laundry': lambda v: 1 if (isinstance(v, (int, float)) and v < 2) else 0,
                        'transportation': lambda v: 1 if (isinstance(v, (int, float)) and v < 3) else 0,
                        'medications': lambda v: 1 if (isinstance(v, (int, float)) and v == 0) else 0,
                        'finances': lambda v: 1 if (isinstance(v, (int, float)) and v < 2) else 0,
                    }
                    
                    for category, scoring_func in scoring_rules.items():
                        if category in iadl_json:
                            category_value = iadl_json[category]
                            total_score += scoring_func(category_value)

                    iadl_record['total_score'] = total_score
                else:
                    iadl_record['total_score'] = None
                    self.log_error(f"IADL data is missing for submission {row['study_submission_id']}", row['study_submission_id'])
            except Exception as e:
                self.log_error(f"Error processing IADL data for submission {row['study_submission_id']}: {e}", row['study_submission_id'])
                iadl_record['total_score'] = None

            iadl_data.append(iadl_record)
            
        iadl_df = pd.DataFrame(iadl_data)
        iadl_path = os.path.join(self.RESULTS_DIR, "iadl_questionnaire.csv")
        print(f"Writing IADL questionnaire data to {iadl_path}")
        iadl_df.to_csv(iadl_path, index=False)
        
        # Process health questionnaire
        print("... Processing health questionnaire")
        health_data = []

        for _, row in self.study_submissions.iterrows():
            health_record = {'study_submission_id': row['study_submission_id']}

            try:
                if pd.notna(row['health']):
                    health_json = json.loads(row['health']) if isinstance(row['health'], str) else row['health']

                    # Extract all keys as columns
                    for key, value in health_json.items():
                        health_record[key] = value

                    # Score health questionnaire by counting issues in each section
                    # Section A - Hearing, vision, and speech
                    n_issues_hearing_vision_speech = 0
                    if 'hearingDifficulty' in health_json:
                        val = health_json['hearingDifficulty']
                        if val in ['yes', 'not_sure']:
                            n_issues_hearing_vision_speech += 1
                    if 'visionSeePicture' in health_json:
                        val = health_json['visionSeePicture']
                        if val == 'no':
                            n_issues_hearing_vision_speech += 1
                    if 'visionReadText' in health_json:
                        val = health_json['visionReadText']
                        if val == 'no':
                            n_issues_hearing_vision_speech += 1
                    if 'speechClarity' in health_json:
                        val = health_json['speechClarity']
                        if val in ['yes', 'not_sure']:
                            n_issues_hearing_vision_speech += 1
                    
                    health_record['n_issues_hearing_vision_speech'] = n_issues_hearing_vision_speech

                    # Section B - Neurological and cognitive health history
                    n_issues_neurological = 0
                    if 'memoryCognitiveDiagnosis' in health_json:
                        val = health_json['memoryCognitiveDiagnosis']
                        if val in ['yes', 'not_sure']:
                            n_issues_neurological += 1
                    if 'strokeTIA' in health_json:
                        val = health_json['strokeTIA']
                        if val in ['yes', 'not_sure']:
                            # Check for ongoing symptoms if strokeTIA is yes
                            if val == 'yes' and 'strokeTIASymptoms' in health_json:
                                symptoms_val = health_json['strokeTIASymptoms']
                                if symptoms_val in ['yes_ongoing', 'not_sure']:
                                    n_issues_neurological += 1
                    if 'headInjury' in health_json:
                        val = health_json['headInjury']
                        if val == 'yes':
                            # Only count if injury occurred less than 5 years ago
                            if 'headInjuryTiming' in health_json:
                                timing_val = health_json['headInjuryTiming']
                                if timing_val in ['within_12_months', '1_5_years', 'not_sure']:
                                    n_issues_neurological += 1
                            else:
                                # If timing is missing but headInjury is yes, count as issue
                                n_issues_neurological += 1
                    if 'ongoingBrainSymptoms' in health_json:
                        val = health_json['ongoingBrainSymptoms']
                        if val in ['yes', 'not_sure']:
                            n_issues_neurological += 1
                    if 'otherNeurologicalConditions' in health_json:
                        val = health_json['otherNeurologicalConditions']
                        # Handle both list/array and single value formats
                        if isinstance(val, list):
                            # If 'none' is in the list, no issues; otherwise, if list has any items, it's an issue
                            if len(val) > 0 and 'none' not in val:
                                n_issues_neurological += 1
                        elif val is not None and val != 'none':
                            n_issues_neurological += 1
                    if 'speechLanguageDiagnoses' in health_json:
                        val = health_json['speechLanguageDiagnoses']
                        # Handle both list/array and single value formats
                        if isinstance(val, list):
                            # If 'none' is in the list, no issues; otherwise, if list has any items, it's an issue
                            if len(val) > 0 and 'none' not in val:
                                n_issues_neurological += 1
                        elif val is not None and val != 'none':
                            n_issues_neurological += 1
                    
                    health_record['n_issues_neurological'] = n_issues_neurological

                    # Section C - Mental health and current state
                    n_issues_mental_health = 0
                    if 'seriousPsychiatricConditions' in health_json:
                        val = health_json['seriousPsychiatricConditions']
                        if val in ['yes', 'not_sure']:
                            n_issues_mental_health += 1
                    if 'currentSevereSymptoms' in health_json:
                        val = health_json['currentSevereSymptoms']
                        if val in ['yes', 'not_sure']:
                            n_issues_mental_health += 1
                    if 'substancesToday' in health_json:
                        val = health_json['substancesToday']
                        if val == 'yes':
                            n_issues_mental_health += 1
                    
                    health_record['n_issues_mental_health'] = n_issues_mental_health

                    # Section D - Medication and acute health
                    n_issues_medication = 0
                    if 'sedatingMedications' in health_json:
                        val = health_json['sedatingMedications']
                        if val in ['yes', 'not_sure']:
                            n_issues_medication += 1
                    if 'acuteIllness' in health_json:
                        val = health_json['acuteIllness']
                        if val == 'yes':
                            n_issues_medication += 1
                    
                    health_record['n_issues_medication'] = n_issues_medication
                    
                    # Calculate total issues (sum of all specific issue counts)
                    health_record['n_issues'] = (
                        n_issues_hearing_vision_speech + 
                        n_issues_neurological + 
                        n_issues_mental_health + 
                        n_issues_medication
                    )
                else:
                    health_record['n_issues_hearing_vision_speech'] = None
                    health_record['n_issues_neurological'] = None
                    health_record['n_issues_mental_health'] = None
                    health_record['n_issues_medication'] = None
                    health_record['n_issues'] = None
                    self.log_error(f"Health data is missing for submission {row['study_submission_id']}", row['study_submission_id'])
            except Exception as e:
                self.log_error(f"Error processing health data for submission {row['study_submission_id']}: {e}", row['study_submission_id'])
                health_record['n_issues_hearing_vision_speech'] = None
                health_record['n_issues_neurological'] = None
                health_record['n_issues_mental_health'] = None
                health_record['n_issues_medication'] = None
                health_record['n_issues'] = None

            health_data.append(health_record)

        health_df = pd.DataFrame(health_data)
        health_path = os.path.join(self.RESULTS_DIR, "health_questionnaire.csv")
        print(f"Writing health questionnaire data to {health_path}")
        health_df.to_csv(health_path, index=False)
        
        print("Done preparing questionnaire data.")

    def prepare_data(self):
        print("Starting data preparation:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.load_raw_data()
        self.delete_unnecessary_raw_audio_files()
        self.prepare_id_mappings()
        self.prepare_prolific_data()
        self.prepare_study_submission_data()
        self.create_feedback_file()
        self.check_demographics_consistency()
        self.prepare_demographics()
        self.prepare_questionnaire_data()
        self.descriptive_statistics()
        self.prepare_cognitive_scores()

        self.copy_recorded_data()

        self.check_moca_submission()

        self._run_extra_logic_processor('after_prepare_data')

        self.run_speech_recognition()

        with open(os.path.join(self.RESULTS_DIR, "..", "_status.txt"), "a") as f:
            f.write(f"Data preparatation: Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

