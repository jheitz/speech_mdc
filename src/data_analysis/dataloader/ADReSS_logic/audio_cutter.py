import os.path

from pydub import AudioSegment
import pandas as pd
import numpy as np

from util.helpers import create_directory
from data_analysis.util.decorators import cache_to_file_decorator

class AudioCutter:
    """
    Cut audio files based on known segments to only include parts where patient is speaking
    """

    def __init__(self, config, constants, run_parameters=None):
        self.config = config
        self.CONSTANTS = constants
        self.run_parameters = run_parameters

        self.name = "Audio Cutter"
        self.store_segments_to_disk = True
        print(f"Initializing {self.name}")

    def _complete_segmentation(self, segmentation_df):
        """
        Sometimes, the segmentation file only ras rows for the interviewer (INV). The rest should be PAR.
        This method takes the segmentation dataframe and introduces PAR sections accordingly
        """
        # all time boundaries (beginning / ends)
        time_boundaries = np.concatenate((np.array([0]), segmentation_df['begin'], segmentation_df['end']))
        time_boundaries_sorted = np.sort(np.unique(time_boundaries))

        # create rows for each segment
        rows = np.vstack((time_boundaries_sorted[:-1], time_boundaries_sorted[1:])).T
        rows_df = pd.DataFrame(rows, columns=['begin', 'end'])

        # join to existing data
        new_segmentation = segmentation_df.merge(rows_df, on=['begin', 'end'], how="right")

        # fill speaker = PAR where null (not in original df)
        new_segmentation['speaker'] = np.where(~new_segmentation['speaker'].isna(), new_segmentation['speaker'], "PAR")

        return new_segmentation

    def cut_to_participant_segments(self, audio_file_path, segmentation, new_path, segments_directory=None):
        """
        Keep only the parts of an audio file in which the participant speaks and write the result to
        new_path. segmentation is a dataframe with the columns speaker / begin / end (in milliseconds);
        everything that it does not explicitly attribute to the interviewer counts as participant,
        see _complete_segmentation. If segments_directory is given, the individual segments are
        written there as well, for debugging.
        """
        # importing file from location by giving its path
        sound = AudioSegment.from_file(audio_file_path, format="wav")

        # complete potentially missing PAR rows in segmentation file
        segmentation = self._complete_segmentation(segmentation)
        # get only PAR segments (from participant, not interviewer)
        segmentation = segmentation.query("speaker == 'PAR'")

        # extract the actual audio segments
        segments = [sound[seg['begin']:seg['end']] for i, seg in segmentation.iterrows()]
        assert len(segments) > 0, f"No participant segments in the segmentation of {audio_file_path}"

        # store segments to disk, for debugging
        if segments_directory is not None:
            create_directory(segments_directory)
            segments_path = lambda i, file: os.path.join(segments_directory, f"{file}_{i}.wav")
            [seg.export(segments_path(i, os.path.basename(audio_file_path)), format="wav") for i, seg in enumerate(segments)]

        # sum these to get a new entire audio
        new_audio = sum(segments)

        # Saving file in required location
        create_directory(os.path.dirname(new_path))
        new_audio.export(new_path, format="wav")

        return new_path

    @cache_to_file_decorator()
    def cut(self, audio_file_path, segments_file_path, target_directory):
        # segmentation df
        segmentation = pd.read_csv(segments_file_path)

        create_directory(target_directory)
        new_path = os.path.join(target_directory, os.path.basename(audio_file_path))
        segments_directory = os.path.join(target_directory, "segments") if self.store_segments_to_disk else None

        return self.cut_to_participant_segments(audio_file_path, segmentation, new_path, segments_directory)


    def preprocess_dataset(self, dataset: AudioDataset) -> AudioDataset:
        raise ValueError("Not complete (check original repo), only the cut_to_participant_segments method is used here")

