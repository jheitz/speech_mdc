import os
import pandas as pd
import numpy as np
import re
import logging
import sys
import librosa
import time
from datetime import datetime

sys.path.insert(0, '..') # to make the import from parent dir work

from data_preparation.test_scoring.fluency.string_alignment import NeedlemanWunsch

from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters


class DataQualityChecker:
    def __init__(self, run_parameters: RunParameters, config: Config, CONSTANTS: Constants):
        self.run_parameters = run_parameters
        self.config = config
        self.CONSTANTS = CONSTANTS

        # results / preprocessed dir
        try:
            self.RESULTS_DIR = os.path.join(run_parameters.results_dir, "data")
        except:
            self.RESULTS_DIR = os.path.join(CONSTANTS.DATA_PROCESSED_2026, config.name, "data")
        assert os.path.exists(self.RESULTS_DIR)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if run_parameters.logfile is not None:
            handler = logging.FileHandler(run_parameters.logfile, encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)


    def load_data(self):
        self.acs_outcomes = pd.read_csv(os.path.join(self.RESULTS_DIR, "acs_outcomes.csv"))
        self.prolific_data = pd.read_csv(os.path.join(self.RESULTS_DIR, "prolific_data.csv"))
        self.study_submissions = pd.read_csv(os.path.join(self.RESULTS_DIR, "study_submissions.csv"))

    def check_number_of_data_rows(self):
        print("Checking number of data rows...")
        n_submissions = self.study_submissions.shape[0]
        n_prolific = self.prolific_data.shape[0]
        if n_prolific != n_submissions:
            self.logger.error(f"Prolific data has other number of rows than study_submissions: {n_prolific} vs. {n_submissions}")

        n_acs_outcomes = self.acs_outcomes.shape[0]
        if n_acs_outcomes != n_submissions:
            self.logger.error(f"ACS outcomes data has other number of rows than study_submissions: {n_acs_outcomes} vs. {n_submissions}")

    def check_audio_files(self):
        print("Checking audio files...")
        expected_audio_files = ['check.wav', 'cookieTheft.wav', 'cookieTheftImmediateRecall.wav',
                                'cookieTheftDelayedRecall.wav', 'journaling.wav', 'phonemicFluencyF.wav',
                                'phonemicFluencyA.wav', 'phonemicFluencyS.wav',
                                'picnicScene.wav', 'pictureNaming.wav', 'semanticFluency.wav',
                                'semanticFluencyVegetables.wav', 'storyImmediateRecall.wav', 'storyDelayedRecall.wav']
        for id in self.study_submissions.study_submission_id:
            audio_dir = os.path.join(self.RESULTS_DIR, str(id), 'audio')
            missing_files = [file for file in expected_audio_files if not os.path.exists(os.path.join(audio_dir, file))]
            if len(missing_files) > 0:
                self.logger.warning(f"Missing audio files {missing_files} for submission {id}.")

    def _get_audio_raw_ids(self):
        with open(os.path.join(self.RESULTS_DIR, "..", "1_stdout.txt"), "r") as f:
            log_lines = f.readlines()
            match_pattern = r"Conversion complete. The file '(.*?)' has been written \(source: (.*?)/p/assets/public/([0-9]+).webm\)"
            audio_raw_ids = {}
            for line in log_lines:
                match = re.search(match_pattern, line)
                if match:
                    audio_file = os.path.abspath(match.group(1))
                    raw_id = match.group(3)
                    if audio_file in audio_raw_ids:
                        raise ValueError(f"Unclear raw audio file id for {audio_file} in data preprocessor log file? Found both {audio_raw_ids[audio_file]} and {raw_id}. This is a problem because this ID is used to delete audio files in extra logic")
                    audio_raw_ids[audio_file] = raw_id
            return audio_raw_ids

    def check_audio_durations(self):
        print("Checking audio durations")

        audio_raw_ids = self._get_audio_raw_ids()

        expected_audio_durations = {
            'check': [1, 30],
            'cookieTheft': [45, 10*60],
            'cookieTheftImmediateRecall': [45, 10*60],
            'cookieTheftDelayedRecall': [45, 10*60],
            'picnicScene': [45, 10*60],
            'phonemicFluencyF': [58, 62],
            'phonemicFluencyA': [58, 62],
            'phonemicFluencyS': [58, 62],
            'semanticFluency': [58, 62],
            'semanticFluencyVegetables': [58, 62],
            'pictureNaming': [30, 10*60],
            'journaling': [45, 10*60],
            'storyImmediateRecall': [20, 90],
            'storyDelayedRecall': [20, 90],
            'moca-naming': [5, 40],
            'moca-memory-trial1': [0, 10*60],
            'moca-memory-trial2': [0, 10*60],
            'moca-attention-forward': [1, 15],
            'moca-attention-backward': [1, 15],
            'moca-attention-serial7': [10, 90],
            'moca-sentence-repetition-1': [3, 20],
            'moca-sentence-repetition-2': [3, 20],
            'moca-delayed-recall': [3, 60],
        }
        for id in self.study_submissions.study_submission_id:
            id_dir = os.path.join(self.RESULTS_DIR, str(id))
            
            # Check non-MoCA audio durations
            audio_durations_path = os.path.join(id_dir, 'audio_durations.csv')
            if os.path.exists(audio_durations_path):
                audio_durations = pd.read_csv(audio_durations_path)
                audio_durations = audio_durations.dropna()
                audio_durations['full_path'] = audio_durations['task'].map(lambda task: os.path.abspath(os.path.join(id_dir, 'audio', f"{task}.wav")))
                audio_durations['raw_id'] = audio_durations['full_path'].map(lambda full_path: audio_raw_ids.get(full_path, "unknown"))
                invalid_durations = [row for idx, row in audio_durations.iterrows() if not (expected_audio_durations[row.task][0] <= row.duration <= expected_audio_durations[row.task][1])]
                if len(invalid_durations) > 0:
                    self.logger.warning(f"Submission {id} has {len(invalid_durations)} audio files with invalid durations: " + \
                                    ", ".join([f"{row.task}: {row.duration} [raw id: {row.raw_id}] (should be {expected_audio_durations[row.task]})" for row in invalid_durations]))
            
            # Check MoCA audio durations
            moca_audio_durations_path = os.path.join(id_dir, 'moca', 'audio_durations.csv')
            if os.path.exists(moca_audio_durations_path):
                moca_audio_durations = pd.read_csv(moca_audio_durations_path)
                moca_audio_durations = moca_audio_durations.dropna()
                moca_audio_durations['full_path'] = moca_audio_durations['task'].map(lambda task: os.path.abspath(os.path.join(id_dir, 'moca', f"{task}.wav")))
                moca_audio_durations['raw_id'] = moca_audio_durations['full_path'].map(lambda full_path: audio_raw_ids.get(full_path, "unknown"))
                invalid_moca_durations = [row for idx, row in moca_audio_durations.iterrows() if not (expected_audio_durations[row.task][0] <= row.duration <= expected_audio_durations[row.task][1])]
                if len(invalid_moca_durations) > 0:
                    self.logger.warning(f"Submission {id} has {len(invalid_moca_durations)} MoCA audio files with invalid durations: " + \
                                    ", ".join([f"{row.task}: {row.duration} [raw id: {row.raw_id}] (should be {expected_audio_durations[row.task]})" for row in invalid_moca_durations]))


    def _get_transcripts_info(self, submission_id):
        transcripts_audio = pd.read_csv(os.path.join(self.RESULTS_DIR, str(submission_id), 'audio', 'ASR', 'transcriptions.csv'))
        transcript_dir_moca = os.path.join(self.RESULTS_DIR, str(submission_id), 'moca', 'ASR')
        try:  # fails if no MoCA data i.e. empty transcriptions.csv
            transcripts_moca = pd.read_csv(os.path.join(transcript_dir_moca, 'transcriptions.csv'))
            transcript_info = pd.concat((
                transcripts_audio,
                transcripts_moca
            ), ignore_index=True)
        except Exception as e:
            transcript_info = transcripts_audio
        return transcript_info

    def check_silence(self):
        print("Checking VAD / silences in audio files")
        audio_raw_ids = self._get_audio_raw_ids()
        for id in self.study_submissions.study_submission_id:
            transcript_info = self._get_transcripts_info(id)

            # load audio durations when applied VAD (e.g. removed silence)
            VAD_file_dir = os.path.join(self.CONSTANTS.CACHE_DIR, "audio_VAD")
            VAD_audio_durations = []
            for file in os.listdir(VAD_file_dir):
                try:
                    task, submission, filetype = re.match(r"(.*)_([0-9]+)_VAD\.([a-zA-Z]{3,})", file).group(1,2,3)
                except AttributeError as e:
                    continue
                if filetype == 'wav':
                    audio_duration_VAD = librosa.get_duration(path=os.path.join(VAD_file_dir, file))
                    VAD_audio_durations.append({'task': task, 'submission': int(submission), 'VAD_audio_duration': audio_duration_VAD})

            VAD_audio_durations_df = pd.DataFrame(VAD_audio_durations)

            transcript_info_extended = transcript_info.merge(VAD_audio_durations_df, on=["submission", 'task'], how="left")
            # if VAD_audio_duration is None, this means that the VAD file does not exist, i.e. there were no non-silence segments detected
            transcript_info_extended['VAD_audio_duration'] = transcript_info_extended['VAD_audio_duration'].fillna(0)
            transcript_info_extended['fraction_of_silence'] = (transcript_info_extended['audio_duration'] - transcript_info_extended['VAD_audio_duration']) / transcript_info_extended['audio_duration']
            subdir = lambda task_name: 'moca' if task_name.startswith('moca-') else 'audio'
            transcript_info_extended['full_path'] = transcript_info_extended[["submission", 'task']].apply(lambda row: os.path.join(self.RESULTS_DIR, str(row['submission']), subdir(row['task']), str(row['task']) + '.wav'), axis=1)
            transcript_info_extended['raw_id'] = transcript_info_extended['full_path'].map(lambda full_path: audio_raw_ids.get(full_path, "unknown"))

            spontaneous_speech_extensive_silence = (transcript_info_extended['fraction_of_silence'] > 0.5) & (transcript_info_extended['task'].isin(['journaling', 'picnicScene', 'cookieTheft', 'cookieTheftImmediateRecall', 'cookieTheftDelayedRecall']))
            language_task_extensive_silence = (transcript_info_extended['fraction_of_silence'] > 0.8) & (transcript_info_extended['task'].isin(['phonemicFluencyF', 'phonemicFluencyA', 'phonemicFluencyS', 'semanticFluency', 'semanticFluencyVegetables', 'pictureNaming', 'storyImmediateRecall', 'storyDelayedRecall']))
            extensive_silence = transcript_info_extended[spontaneous_speech_extensive_silence | language_task_extensive_silence]
            if extensive_silence.shape[0] > 0:
                extensive_silence_desc = ", ".join([f'{task} ({frac*100:.1f}%, raw_id: {raw_id})' for task, frac, raw_id in zip(extensive_silence.task.to_list(), extensive_silence.fraction_of_silence.to_list(), extensive_silence.raw_id.to_list())])
                self.logger.warning(f"Extensive silence for submission {extensive_silence.submission.iloc[0]}: {extensive_silence_desc}")

    def check_repeated_audio_chunks(self):
        # check if the first audio chunk is repeated, i.e. if the signal repeats after ~10sec
        print("Checking for repeated audio chunks...")
        for id in self.study_submissions.study_submission_id:
            audio_dir = os.path.join(self.RESULTS_DIR, str(id), 'audio')
            for filename in os.listdir(audio_dir):
                if filename.endswith("wav") and not filename.startswith("check"):
                    audio_file_path = os.path.join(audio_dir, filename)
                    signal, sample_rate = librosa.load(audio_file_path, sr=1000)

                    window_size_seconds = 8
                    window_size = sample_rate * window_size_seconds
                    first_10sec = signal[:window_size]

                    norm_first_10sec = np.linalg.norm(first_10sec)
                    def vector_similarity(window):
                        return np.dot(window, first_10sec) / (np.linalg.norm(window) * norm_first_10sec)

                    start_obervation_sec = 9
                    end_obervation_sec = 11
                    relevant_signal_section = signal[int(sample_rate * start_obervation_sec):int(sample_rate * (end_obervation_sec + window_size_seconds))]
                    similarity = pd.Series(relevant_signal_section).rolling(window=window_size).apply(vector_similarity)
                    max_similarity = np.max(np.abs(similarity))
                    if max_similarity > 0.5:
                        self.logger.error(f"Repeated audio chunk for {audio_file_path}?")

            #print(f" ({(time.time() - start_time)}s)")

    def check_transcripts(self):
        print("Checking transcripts for audio files")
        for id in self.study_submissions.study_submission_id.sort_values():
            transcript_info = self._get_transcripts_info(id)
            transcript_info = transcript_info[~transcript_info['task'].isin(['check', 'moca-attention-serial7', 'moca-attention-forward', 'moca-attention-backward'])]  # exclude tasks with numbers only, they can be transcribed differently

            # align google and whisper transcription, to check difference as a metric of quality
            string_aligner = NeedlemanWunsch()
            def preprocess_transcription_for_alignment(transcript):
                try:
                    transcript = re.sub(r'\s+', ' ', transcript)
                    transcript = re.sub('[^A-Za-z0-9 ]+', '', transcript)
                except:
                    pass
                return transcript
            def string_distance_metric(row):
                whisper = preprocess_transcription_for_alignment(row.text_whisper)
                google = preprocess_transcription_for_alignment(row.text_google)
                try:
                    _, _, rx, ry, counts = string_aligner.align_strings(whisper, google, return_details=True)
                except:
                    return pd.Series([None, None])
                total_count = sum([counts[c] for c in counts if c != 'match'])
                frac = total_count / max(len(row.text_whisper), len(row.text_google))
                return pd.Series([total_count, frac])
            transcript_info[['transcript_distance', 'transcript_distance_rel']] = transcript_info.apply(string_distance_metric, axis=1)

            large_transcript_diff = transcript_info[
                (transcript_info['task'].isin(self.CONSTANTS.LUHA_2026_SPONATANEOUS_SPEECH_TASKS) & (transcript_info.transcript_distance_rel > 0.2)) |
                (transcript_info['task'].isin(self.CONSTANTS.LUHA_2026_LANGUAGE_TASKS) & (transcript_info.transcript_distance_rel > 0.3))
            ]
            if large_transcript_diff.shape[0] > 0:
                extensive_silence_desc = ", ".join([f'{task} ({frac*100:.1f}% difference)' for task, frac in zip(large_transcript_diff.task.to_list(), large_transcript_diff.transcript_distance_rel.to_list())])
                self.logger.warning(f"Large difference between transcripts (low quality?) for submission {large_transcript_diff.submission.iloc[0]}: {extensive_silence_desc}")

    def check_spontaneous_speech_topics(self):
        """Check if spontaneous speech transcripts match the expected topic."""
        print("Checking spontaneous speech topics...")
        
        # Define keywords for each topic
        topic_keywords = {
            'cookie_theft': ['cookie', 'sink', 'overspilling', 'overflowing', 'kitchen', 'dishes'],
            'picnic_scene': ['picnic', 'kite', 'book', 'dog', 'wine', 'sand castle'],
            'story': ['robbed', 'mugged', 'police', 'boston', 'school', 'cafeteria'],
            'journaling': ['last week', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'shopping', 'visit']
        }
        
        # Map tasks to expected topics
        task_to_expected_topic = {
            'cookieTheft': 'cookie_theft',
            'cookieTheftImmediateRecall': 'cookie_theft',
            'cookieTheftDelayedRecall': 'cookie_theft',
            'picnicScene': 'picnic_scene',
            'journaling': 'journaling',
            'storyImmediateRecall': 'story',
            'storyDelayedRecall': 'story'
        }
        
        for id in self.study_submissions.study_submission_id.sort_values():
            transcript_info = self._get_transcripts_info(id)
            
            for task_name, expected_topic in task_to_expected_topic.items():
                # Get transcript for this task
                task_transcript = transcript_info[transcript_info['task'] == task_name]
                
                if task_transcript.empty:
                    continue  # Task not found, skip (may be handled by other checks)
                
                # Get Google transcript
                if 'text_google' not in task_transcript.columns or pd.isna(task_transcript.iloc[0]['text_google']):
                    continue  # No Google transcript available
                
                transcript_text = str(task_transcript.iloc[0]['text_google']).lower()
                
                # Count keyword matches for each topic
                topic_match_counts = {}
                topic_found_keywords = {}
                
                for topic, keywords in topic_keywords.items():
                    found_keywords = [kw for kw in keywords if kw in transcript_text]
                    topic_match_counts[topic] = len(found_keywords)
                    topic_found_keywords[topic] = found_keywords
                
                # Find the topic with the most keyword matches (most likely topic)
                max_match_count = max(topic_match_counts.values())
                
                if max_match_count == 0:
                    # No keywords found from any topic
                    self.logger.error(f"Submission {id}, task {task_name}: Expected topic '{expected_topic}', but no keywords from any topic were found")
                else:
                    expected_count = topic_match_counts[expected_topic]
                    
                    # Check if the expected topic has fewer matches than the maximum
                    # If it has the same number of matches (tie), that's fine
                    if expected_count < max_match_count:
                        # Find the topic(s) with the maximum count
                        most_likely_topics = [topic for topic, count in topic_match_counts.items() if count == max_match_count]
                        most_likely_topic = most_likely_topics[0]  # Pick first one for reporting
                        
                        found_keywords_desc = ", ".join([f"'{kw}'" for kw in topic_found_keywords[most_likely_topic]])
                        error_msg = (f"Submission {id}, task {task_name}: Expected topic '{expected_topic}' "
                                    f"({expected_count} keyword matches), but most likely topic is '{most_likely_topic}' "
                                    f"({max_match_count} keyword matches: {found_keywords_desc})")
                        self.logger.error(error_msg)

    def check_google_transcript_length(self):
        """Check if Google transcripts have at least 10 words for specified tasks."""
        print("Checking Google transcript word counts...")
        
        tasks_to_check = [
            'cookieTheft',
            'cookieTheftImmediateRecall',
            'cookieTheftDelayedRecall',
            'journaling',
            'picnicScene',
            'storyImmediateRecall',
            'storyDelayedRecall'
        ]
        
        for id in self.study_submissions.study_submission_id.sort_values():
            transcript_info = self._get_transcripts_info(id)
            
            for task_name in tasks_to_check:
                # Get transcript for this task
                task_transcript = transcript_info[transcript_info['task'] == task_name]
                
                if task_transcript.empty:
                    continue  # Task not found, skip (may be handled by other checks)
                
                # Get Google transcript
                if 'text_google' not in task_transcript.columns or pd.isna(task_transcript.iloc[0]['text_google']):
                    self.logger.error(f"Submission {id}, task {task_name}: Google transcript is missing or empty")
                    continue
                
                transcript_text = str(task_transcript.iloc[0]['text_google']).strip()
                
                # Count words (split by whitespace and filter out empty strings)
                word_count = len([word for word in transcript_text.split() if word.strip()])
                
                if word_count < 10:
                    self.logger.error(f"Submission {id}, task {task_name}: Google transcript has only {word_count} words (minimum required: 10)")

    def check(self):
        print("Starting data quality check:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.load_data()
        self.check_number_of_data_rows()
        self.check_audio_files()
        self.check_audio_durations()
        self.check_silence()
        self.check_repeated_audio_chunks()
        self.check_transcripts()
        self.check_spontaneous_speech_topics()
        self.check_google_transcript_length()

        with open(os.path.join(self.RESULTS_DIR, "..", "_status.txt"), "a") as f:
            f.write(f"Data quality check: Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

