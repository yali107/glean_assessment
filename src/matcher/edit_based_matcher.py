import py_stringmatching as sm

from src.matcher.base_matcher import BaseMatcher


class EditMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()

    def jaro_winkler_score(self, str_pair, sim_score=True):
        """
        calculate jaro winkler similarity between two strings
        :return: similarity score or raw score (0 to 1)
        """
        s1, s2 = self._check_input(str_pair)
        jaro_wink = sm.JaroWinkler()
        return jaro_wink.get_sim_score(s1, s2) if sim_score else jaro_wink.get_raw_score(s1, s2)

    def lev_score(self, str_pair, sim_score=True):
        """
        calculate levenshtein similarity between two strings
        :return: similarity score or raw score (0 to 1)
        """
        s1, s2 = self._check_input(str_pair)
        lev = sm.Levenshtein()
        return lev.get_sim_score(s1, s2) if sim_score else lev.get_raw_score(s1, s2)

    def hamming_score(self, str_pair, sim_score=True):
        """
        calculate hamming similarity between two strings
        :return: similarity score or raw score
        """
        s1, s2 = self._check_input(str_pair)
        if len(s1) != len(s2):
            return 0
        hamming = sm.HammingDistance()
        return hamming.get_sim_score(s1, s2) if sim_score else hamming.get_raw_score(s1, s2)

# matcher = EditMatcher()
# print(matcher.lev_score(['abc def', 'ABC DEF'.lower()]))
