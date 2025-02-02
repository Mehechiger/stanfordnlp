"""
Utility functions for data transformations.
"""

import torch

from stanfordnlp.models.common.combined import NO_LABEL
import stanfordnlp.models.common.seq2seq_constant as constant

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size, pad_id=constant.PAD_ID):
    """ Convert (list of )+ tokens to a padded LongTensor. """
    if type(tokens_list[0]) == bool: return torch.LongTensor(tokens_list)  # has_tag or has_syn

    sizes = []
    x = tokens_list
    while isinstance(x[0], list):
        sizes.append(max(len(y) for y in x))
        x = [z for y in x for z in y]
    tokens = torch.LongTensor(batch_size, *sizes).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        if NO_LABEL in s:  # Checks and fills all NO_LABEL with pad_id which will be masked out later.
            for w in s: assert w == NO_LABEL
            s = [pad_id, ] * len(s)
        if len(sizes) == 1:
            tokens[i, :len(s)] = torch.LongTensor(s)
        elif len(sizes) == 2:
            for j, p in enumerate(s): tokens[i, j, :len(p)] = torch.LongTensor(p)
        else:
            raise NotImplementedError
    return tokens

def get_float_tensor(features_list, batch_size):
    if features_list is None or features_list[0] is None:
        return None
    seq_len = max(len(x) for x in features_list)
    feature_len = len(features_list[0][0])
    features = torch.FloatTensor(batch_size, seq_len, feature_len).zero_()
    for i,f in enumerate(features_list):
        features[i,:len(f),:] = torch.FloatTensor(f)
    return features

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
