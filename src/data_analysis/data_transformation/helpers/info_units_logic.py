"""
Information units tagging for Cookie Theft picture description.
Based on Croisile et al. Code by Fraser et al. and adapted by Antonia Popp.
Originally in analyses/kw09/analyzeCookie.py
"""

from util.helpers import safe_divide


class UnitTagger:
    """Class to tag occurrences of information units based on Croisile et al."""

    def __init__(self):
        # Dictionary of keywords for each information unit of Croisile et al.
        # -ing forms included due to lemmatizer problems.
        self.keywords = {
            'steal': ['take', 'taking', 'steal', 'stealing', 'climb', 'climbing', 'sneak', 'sneaking', 'raid', 'raiding',
                      'remove', 'rob', 'robbing', 'crawl', 'crawling', 'swipe', 'swiping', "grab", "grabbing"],

            'fall': ['fall', 'falling', 'fell', 'tumble', 'tumbling', 'crash', 'crashing', 'tip', 'tipping',
                     'hit', 'hitting', 'teeter', 'teetering', 'upend', 'tilt', 'tilting', 'slip', 'slipping',
                     'lose', 'losing', 'topple', 'toppling', 'overturn', 'overturning', 'collapse', 'collapsing', 'tilty',
                     'stumble', 'stumbling', 'slant', 'slanting', 'depart', 'upset', 'upsetting', 'wobble', 'wobbling'],

            'wash': ['wash', 'washing', 'clean', 'cleaning', 'scrub', 'scrubbing', 'dry', 'drying', 'wipe', 'wiping'],

            'overflow': ['overflow', 'overflowing', 'flow', 'flowing', 'flood', 'flooding',
                         'run', 'running', 'spill', 'pour', 'pouring', 'overspilling', 'spilling', 'drip', 'dripping',
                         'splash', 'splashing', 'overrun', 'overrunning', 'overran', 'overflown', 'overfly',
                         'gush', 'gushing', 'cascade', 'cascading', 'leak', 'leaking', 'sprinkle', 'sprinkling',
                         'splatter', 'splattering', 'clog', 'drain', 'slop', 'slopping', 'spread', 'spreading'],

            'girlaction': ['shush', 'shushing', 'hush', 'hushing', 'lip', 'mouth', 'nose', 'laugh',
                           'laughing', 'hand', 'finger', 'eat', 'eating', 'point', 'pointing', 'receive', 'receiving',
                           'upraise', 'wait', 'waiting', 'shh', 'giggle', 'giggling', 'raise', 'raising',
                           'touch', 'touching', 'caution', 'cautioning', 'gesture', 'gesturing', 'pick', 'picking',
                           'signal', 'whisper', 'whispering', 'admonish', 'admonishing', 'beg', 'begging',
                           'snicker', 'snickering', 'yell', 'yelling', 'cry', 'crying', 'talk', 'talking',
                           'ask', 'asking', 'criticize', 'criticizing', 'help', 'helping', 'wave', 'waving', 'coax', 'coaxing'],

            'indifference': ['indifferent', 'notice', 'oblivious', 'aware', 'unaware', 'attention', 'distract',
                             'daydream', 'dream', 'forget', 'back', 'watch', 'engross', 'daze', 'depress', 'depression',
                             'turn', 'turned', 'care', 'neglect', 'neglecting', 'daft', 'overlook', 'preoccupy',
                             'blind', 'disturb', 'immerse', 'immersed', 'ignore', 'ignoring'],

            # SUBJECTS
            'boy': ['boy', 'son', 'brother', 'guy', 'lad', 'johnny', 'billy', 'junior', 'nephew', 'grandson', "male", "he"],
            'girl': ['girl', 'daughter', 'sister', 'gal', 'sissy', 'girlfriend'],
            'woman': ['female', 'woman', 'adult', 'grownup', 'mother', 'lady', 'mom', 'mama', 'momma', 'mommy', 'wife', 'ma', 'housewife'],

            # OBJECTS
            'cookie': ['cookie', 'biscuit', 'cake', 'treat', 'cooky', 'goody'],
            'jar': ['jar', 'container', 'crock', 'pot', 'cookiejar', 'cookeiejar', 'box', 'tin'],
            'stool': ['stool', 'seat', 'chair', 'ladder', 'stepstool', 'tripod', 'bench', 'stepladder', 'footstool', 'highchair'],
            'sink': ['sink', 'basin', 'washbasin', 'washbowl', 'washstand', 'tap', 'faucet', 'spigot', 'dishsink'],
            'dishcloth': ['dishcloth', 'dishrag', 'towel', 'rag', 'cloth'],
            'water': ['water', 'dishwater', 'liquid', 'puddle', 'pool'],
            'window': ['window', 'frame', 'glass', 'pane', 'sill', 'windowsill'],
            'cupboard': ['cupboard', 'closet', 'shelf', 'cabinet'],
            'dish': ['dish', 'cup', 'plate', 'plat', 'saucer', 'bowl', 'platter', 'mug'],
            'curtain': ['curtain', 'drape', 'drapery', 'blind', 'screen', 'valance', 'shade'],
            'counter': ['counter', 'countertop', 'table', 'tabletop'],

            # PLACES
            'kitchen': ['kitchen', 'room'],
            'exterior': ['outside', 'yard', 'outdoors', 'backyard', 'garden', 'driveway', 'path', 'pathway', 'tree', 'bush',
                         'grass', 'shrubbery', 'shrub', 'wing', 'country', 'garage', 'house', 'lawn', 'walk', 'walkway',
                         'flower', 'leaf', 'landscaping', 'landscape', 'hedge', 'evergreen', 'plant', 'sidewalk', 'roof',
                         'view', 'weed', 'lane', 'foliage', 'extension', 'scenery', 'bloom']
        }

        # Second level rules: ambiguous verb + actor info unit (distance 6)
        self.secondlevel_rules = {
            "steal": [["reach", "reaching", "get", "getting"], "boy"],
            "girlaction": [["reach", "reaching"], "girl"]
        }

    def firstlevel_tag(self, token_list):
        """Tag information units given the lemmatized picture description."""
        tagged = []
        for word in token_list:
            detected_units = []
            word_text = word.text.lower() if word.text else ""
            word_lemma = word.lemma.lower() if word.lemma else ""
            for key in self.keywords:
                if word_text in self.keywords[key] or word_lemma in self.keywords[key]:
                    detected_units.append(key)
            tagged.append(detected_units)
        return tagged

    def secondlevel_tag(self, tagged, token_list):
        """Tag additional units using second-level rules (actor disambiguation)."""
        newtag = []
        last_units = [[], [], [], [], [], []]
        assert len(tagged) == len(token_list)
        for i in range(len(tagged)):
            token = token_list[i]
            current_units = tagged[i]
            newunits = []
            token_text = token.text.lower() if token.text else ""
            token_lemma = token.lemma.lower() if token.lemma else ""
            for tag_rule in self.secondlevel_rules:
                keywords = self.secondlevel_rules[tag_rule][0]
                actor = self.secondlevel_rules[tag_rule][1]
                if token_text in keywords or token_lemma in keywords:
                    for unit_list in last_units:
                        if actor in unit_list:
                            newunits.append(tag_rule)
            newtag.append(newunits)
            last_units.append(current_units)
            last_units.pop(0)
        return newtag

    def unit_counts(self, firstlevel, secondlevel):
        """Combine detected units to counts per information unit."""
        counts = {unit: 0 for unit in self.keywords}
        all_tags = firstlevel + secondlevel
        for detected_units in all_tags:
            for unit in detected_units:
                counts[unit] += 1
        return counts

    def count_summary(self, unit_counts):
        """Calculate total and unique information unit count."""
        total = 0
        unique = 0
        for key in unit_counts:
            if unit_counts[key] > 0:
                total += unit_counts[key]
                unique += 1
        return total, unique

    def full_tagging(self, flat_doc):
        """
        Full tagging pipeline. Returns (unique_count, conciseness_index).
        Conciseness index = information units per non-punctuation word.
        """
        first_tag = self.firstlevel_tag(flat_doc)
        second_tag = self.secondlevel_tag(first_tag, flat_doc)
        counts = self.unit_counts(first_tag, second_tag)
        _, unique_count = self.count_summary(counts)

        non_punct = [token for token in flat_doc if token.pos != "PUNCT"]
        n_words = len(non_punct)
        conc_index = safe_divide(unique_count, n_words)

        return unique_count, conc_index
