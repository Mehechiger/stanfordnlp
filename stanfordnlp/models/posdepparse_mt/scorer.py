"""
Utils and wrappers for scoring MT taggerparsers.
"""
from stanfordnlp.models.common.utils import ud_scores

def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for MT tagger - UD parser scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    las, upos = evaluation['LAS'], evaluation['UPOS']
    las_p, las_r, las_f = las.precision, las.recall, las.f1
    upos_p, upos_r, upos_f = upos.precision, upos.recall, upos.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', 'UPOS']]
        print("LAS\tUPOS")
        print("{:.2f}\t{:.2f}".format(*scores))
    return las_p, las_r, las_f, upos_p, upos_r, upos_f

