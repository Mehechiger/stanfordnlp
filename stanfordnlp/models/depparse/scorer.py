"""
Utils and wrappers for scoring parsers.
"""
import logging
from stanfordnlp.models.common.utils import ud_scores

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for UD parser scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['LAS']
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', ]]
        print("LAS")
        print("{:.2f}".format(*scores))
    return p, r, f

