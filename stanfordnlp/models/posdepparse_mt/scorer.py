"""
Utils and wrappers for scoring MT taggerparsers.
"""
import logging
from stanfordnlp.models.common.utils import ud_scores

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def score(system_conllu_file, gold_conllu_file, verbose=True, do_tagging=True, do_parsing=True):
    """ Wrapper for MT tagger - UD parser scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    if do_parsing:
        las = evaluation['LAS']
        las_p, las_r, las_f = las.precision, las.recall, las.f1
    else:
        las_p, las_r, las_f = -1, -1, -1

    if do_tagging:
        upos = evaluation['UPOS']
        upos_p, upos_r, upos_f = upos.precision, upos.recall, upos.f1
    else:
        upos_p, upos_r, upos_f = -1, -1, -1

    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', 'UPOS']]
        logger.info("LAS\tUPOS")
        logger.info("{:.2f}\t{:.2f}".format(*scores))
    return las_p, las_r, las_f, upos_p, upos_r, upos_f

