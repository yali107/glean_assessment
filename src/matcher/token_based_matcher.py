import py_stringmatching as sm

from src.matcher.base_matcher import BaseMatcher


class TokenMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()

    def jac_score(self, str_pair, sim_score=True) -> float:
        """
        calculate jaccard similarity between two single sets of tokens
        :return: similarity score (0 to 1)
        """
        e1, e2 = self._check_input(str_pair, type_=list)
        jac = sm.Jaccard()
        return jac.get_sim_score(e1, e2) if sim_score else jac.get_raw_score(e1, e2)

    def cos_score(self, str_pair, sim_score=True):
        """
        calculate cosine similarity between two single sets of tokens
        :return: similarity score (0 to 1)
        """
        e1, e2 = self._check_input(str_pair, type_=list)
        cos = sm.Cosine()
        return cos.get_sim_score(e1, e2) if sim_score else cos.get_raw_score(e1, e2)

    def monge_elkan_score(self, str_pair, sim_func=sm.JaroWinkler().get_raw_score):
        """
        calculate monge elkan similarity between two single sets of tokens
        :return: raw_score
        """
        e1, e2 = self._check_input(str_pair, type_=list)
        me = sm.MongeElkan(sim_func=sim_func)
        return me.get_raw_score(e1, e2)
