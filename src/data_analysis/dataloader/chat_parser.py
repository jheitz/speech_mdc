import re
import os
import pandas as pd
import numpy as np

from config.constants import Constants

class ChatTranscriptParser:
    """
    Preprocess transcripts in CHAT (.cha) Format to only get the text
    Manual available at https://talkbank.org/manuals/CHAT.html
    """

    def __init__(self, config, constants=None):
        # different configurations to load / preprocess CHAT format differently
        self.config = config

        if constants is None:
            self.CONSTANTS = Constants()
        else:
            self.CONSTANTS = constants

        required_config = ['only_PAR', 'keep_pauses', 'keep_terminators', 'keep_unintelligable_speech']
        assert all([s in self.config for s in required_config]), \
            f"Missing configs: {[s for s in required_config if s not in self.config]}"

        if 'filter_coded_activity' not in self.config:
            # the WLS data transcripts have an explicit coding for start (@Bg: Activity) and end (@Eg: Activity) of the
            # actual picture description, excluding prior interviewer introduction etc.
            self.config['filter_coded_activity'] = False

        self._test_parser()

    def _preprocess_utterance(self, utterance):
        """
        Preprocess individual utterance

        This follows the PittTranscriptDataLoader of the dementia project
        (/git/dementia/src/dataloader/dataloader.py, _preprocess_transcript)
        """
        # remove events such as &=clears:throat
        utterance = re.sub(r"&=[^\s]+", "", utterance)

        # remove complex local events of the form  [^ text]
        utterance = re.sub(r"\[^\.*?\]", "", utterance)

        # remove phonological fragment, fillers, nonwords, Interposed Words (&um, &-um, &+b, etc)
        utterance = re.sub(r"&[^\s]+", "", utterance)

        # remove pauses (.), (..), (...)
        if self.config['keep_pauses']:
            utterance = re.sub(r"\(\.\)", " SHORTPAUSE ", utterance)
            utterance = re.sub(r"\(\.\.\)", " MEDIUMPAUSE ", utterance)
            utterance = re.sub(r"\(\.\.\.\)", " LONGPAUSE ", utterance)
        else:
            utterance = re.sub(r"\(\.+\)", "", utterance)

        # remove special marking of non-complete words (omitted parts) -> get the full word
        # todo: check if this is a good idea
        utterance = re.sub(r"\((.*?)\)", "\\1", utterance)

        # remove omitted words
        utterance = re.sub(r"0[a-zA-Z]+]", "", utterance)

        if not self.config['keep_unintelligable_speech']:
            # remove unintelligable speech sign "xxx"
            utterance = re.sub(r"xxx", "", utterance)

        # remove sign for letter transcription, e.g. make m out of m@l
        utterance = re.sub(r"([a-zA-Z])@l", "\\1", utterance)

        # remove replacement notation (when a participant says something and the annotator marked what it means)
        # e.g. chair [: stool] --> chair
        utterance = re.sub(r"\[:.*?\]", "", utterance)

        # remove retracing, reformulation, fase start without retracing, unclear retraicing etc. [//], [///] etc
        utterance = re.sub(r"\[/.*?\]", "", utterance)

        # remove custom postcodes, eg. [+ jar]
        utterance = re.sub(r"\[\+.*?\]", "", utterance)

        # remove error markings, e.g. [* s:uk]
        utterance = re.sub(r"\[\*.*?\]", "", utterance)

        # make word repetitions explicit: get [x 3] -> get get get (hacky way with explicit counts)
        utterance = re.sub(r"([^\s]+)\s*\[x 2\]", "\\1 \\1", utterance)
        utterance = re.sub(r"([^\s]+)\s*\[x 3\]", "\\1 \\1 \\1", utterance)
        utterance = re.sub(r"([^\s]+)\s*\[x 4\]", "\\1 \\1 \\1 \\1", utterance)
        utterance = re.sub(r"([^\s]+)\s*\[x 5\]", "\\1 \\1 \\1 \\1 \\1", utterance)

        # remove any other [x N] occurance (happening if e.g. "*PAR:    &um &hm [x 3]")
        utterance = re.sub(r"\[x [0-9]+\]", "", utterance)

        # remove special utterances terminators, like trailing off
        utterance = re.sub(r"\+[.?!/]+", "", utterance)

        # remove overlap markers
        utterance = re.sub(r"\[[<>]\]+", "", utterance)
        utterance = re.sub(r"\+<", "", utterance)

        # remove overlap signs <text>
        utterance = re.sub(r"<(.*?)>", "\\1", utterance)

        # remove unibet trascription words e.g. kɪtʃə˞@u
        utterance = re.sub(r"\d+@u", "", utterance)

        # remove quotation marks
        utterance = re.sub(r"\+\"/\.", "", utterance)
        utterance = re.sub(r"\+\"\.", "", utterance)
        utterance = re.sub(r"\+\"", "", utterance)

        # remove any other Special Utterance Terminators (anything starting with +)
        utterance = re.sub(r"\+[^\s]*", "", utterance)

        # remove basic utterance termination markers (. ? !)
        if not self.config['keep_terminators']:
            utterance = re.sub(r"\s+[.?!]\s?", "", utterance)

        # remove special form markers, e.g. xxx@a, bingo@o
        utterance = re.sub(r"@[a-zA-Z:*$]+", "", utterance)

        # remove satellite markers
        utterance = re.sub(r"[‡„]", "", utterance)

        # replace new lines and other spacing by simple space
        utterance = re.sub(r"\s+", " ", utterance)

        # trim
        utterance = utterance.strip()

        return utterance

    def _extract_disfluency_metrics_from_utterance(self, utterance):
        metrics = {}

        # remove events such as &=clears:throat, to not have them count as disfluency in the below regex
        utterance = re.sub(r"&=[^\s]+", "", utterance)

        # count special sign &, indicating phonological fragment, fillers, nonwords, Interposed Words, filled pauses (&um, etc)
        metrics['n_fillers_fragments'] = len(re.findall(r"&([^\s]+)", utterance))

        # count word and phrase repetitions
        metrics['n_repetitions'] = len(re.findall(r"\[/\]", utterance))
        metrics['n_repetitions'] += len(re.findall(r"([^\s]+)\s*\[x 2\]", utterance))
        metrics['n_repetitions'] += len(re.findall(r"([^\s]+)\s*\[x 3\]", utterance))
        metrics['n_repetitions'] += len(re.findall(r"([^\s]+)\s*\[x 4\]", utterance))
        metrics['n_repetitions'] += len(re.findall(r"([^\s]+)\s*\[x 5\]", utterance))

        # count revisions
        metrics['n_revisions'] = len(re.findall(r"\[//\]", utterance))

        # count pauses
        metrics['n_pause_short'] = len(re.findall(r"\(\.\)", utterance))
        metrics['n_pause_medium'] = len(re.findall(r"\(\.\.\)", utterance))
        metrics['n_pause_long'] = len(re.findall(r"\(\.\.\.\)", utterance))
        metrics['n_pauses'] = metrics['n_pause_short'] + metrics['n_pause_medium'] + metrics['n_pause_long']

        return metrics


    def _match_relevant_utterances(self, transcript):
        # match all parts starting with *PAR: (participant speaking) \x15 (timing information \x15
        # \x15 is a special unicode symbol marking the end of the utterance / start of timing information
        # Also collect utterances with *INV and timing information from transcript
        # if relevant, find beginning and end of relevant section (@Bg: Activity, @Eg: Activity)

        # extract relevant section if applicable
        if self.config['filter_coded_activity']:
            # match @Bg: Activity and @Eg: Activity, to only keep the section in between
            activity_parts = re.findall(r"@Bg:\s*Activity(.*?)@Eg:\s*Activity", transcript,
                                        re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if len(activity_parts) > 0:
                transcript = activity_parts[0]
            else:
                # no activity section found, everything
                print("Warning: No @Bg: Activity and @Eg: Activity section found in transcript. Using full transcript.")

        # match (speaker), (transcript), (timing)
        # An utterance ends at the \x15 that introduces its timing, but not every utterance has one
        # (e.g. "*PAR:\twww", untranscribed material). Those have to end at the next speaker or at
        # the next dependent tier (%mor, %wor, ...) instead, otherwise the match runs on to the
        # \x15 of a *later* utterance and swallows the annotation tiers and the following turn.
        # The terminators are the ones of the dementia project, see _preprocess_utterance.
        parts = re.findall(r"\*(PAR|INV):\s*(.*?)(?:\x15([0-9]+_[0-9]+)?\x15?|(?=\*PAR|\*INV|%))",
                           transcript, re.UNICODE | re.MULTILINE | re.DOTALL)

        def parse_timing(timing_string):
            matches = re.findall(r"([0-9]+)_([0-9]+)", timing_string)
            if len(matches) > 0:
                begin, end = matches[0]
                return int(begin), int(end)
            # an utterance without timing information
            return None, None

        parts = [{'speaker': speaker, 'transcript': text,
                  'begin': parse_timing(timing)[0], 'end': parse_timing(timing)[1]}
                 for speaker, text, timing in parts]

        return pd.DataFrame(parts, columns=['speaker', 'transcript', 'begin', 'end'])

    def preprocess_transcript(self, transcript, path):
        utterances = self._match_relevant_utterances(transcript)

        utterances['transcript'] = utterances['transcript'].apply(self._preprocess_utterance)

        # only keep non-empty utterances
        utterances = utterances[utterances.transcript.str.len() > 0]

        if self.config['only_PAR']:
            utterances = utterances.query("speaker == 'PAR'")

        utterances_joined = " ".join(utterances.transcript)
        return utterances_joined

    def extract_disfluency_metrics(self, transcript):
        """
        Extract disfluency metrics based on manual transcripts of parameter speech
        """
        utterances = self._match_relevant_utterances(transcript)

        # disfluency always only from Participant, not Interviewer
        utterances = utterances.query("speaker == 'PAR'")

        utterances_metrics = [self._extract_disfluency_metrics_from_utterance(p.transcript)
                            for idx, p in utterances.iterrows()]

        # combine
        collected = {}
        metric_names = utterances_metrics[0].keys()
        total_disfluency_count = 0
        for m in metric_names:
            sum_over_utterances = sum([p[m] for p in utterances_metrics])
            collected[m] = sum_over_utterances
            total_disfluency_count += sum_over_utterances
        collected['n_disfluencies'] = total_disfluency_count

        return collected

    def extract_speaker_segmentation(self, transcript):
        """
        Extract timing information for speaker segmentation (which speaker speaks at what second)
        This is useful to cut the interviewer part from the audio
        """
        utterances = self._match_relevant_utterances(transcript)

        return utterances

    def _test_preprocess_utterance(self, input, expected_output):
        assert self._preprocess_utterance(input) == expected_output, f"Input: {input} / Expected: {expected_output} / Output: {self._preprocess_utterance(input)}"

    def _test_parser(self):
        """ Add some unit tests here for the preprocessor functions. """
        test_set = [
            ("+< <he he he> [//] he's passin(g) it down to her", "he he he he's passing it down to her"),
            # fillers and phonological fragments are dropped, marker and all
            ("&uh a lad standing on a stool teetering", "a lad standing on a stool teetering"),
            ("the &-um &+s stool is falling", "the stool is falling"),
            # linkages and lengthened syllables are left as they are transcribed
            ("he's getting cookies out_of the s:tool", "he's getting cookies out_of the s:tool"),
            # <> only spans a line break once the utterance has been joined into one line, which is
            # the last step, so an overlap marker across a line break survives (as in the dementia project)
            ("with a <I don't know what\n he got> [//] cookie jar", "with a <I don't know what he got> cookie jar"),
        ]
        [self._test_preprocess_utterance(input, expected) for (input, expected) in test_set]
        return True

