from collections import Counter, OrderedDict

from stanfordnlp.models.common.vocab import BaseVocab, BaseMultiVocab
from stanfordnlp.models.common.vocab import CompositeVocab, VOCAB_PREFIX, EMPTY, EMPTY_ID
from stanfordnlp.models.common.combined import NO_LABEL
from stanfordnlp.models.common.utils import get_prefixes, get_suffixes


class CharVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([c for sent in self.data for w in sent for c in w[self.idx] if w[self.idx] != NO_LABEL])

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

        for unit in self._id2unit: assert not self.is_no_label(unit), f"Error: {NO_LABEL} found in {self.__name__} in which it shouldn't exist!"

class PrefixVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([prefix for sent in self.data for w in sent for prefix in get_prefixes(w[self.idx], 3) if w[self.idx] != NO_LABEL])

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

        for unit in self._id2unit: assert not self.is_no_label(unit), f"Error: {NO_LABEL} found in {self.__name__} in which it shouldn't exist!"


class SuffixVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([suffix for sent in self.data for w in sent for suffix in get_suffixes(w[self.idx], 3) if w[self.idx] != NO_LABEL])

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

        for unit in self._id2unit: assert not self.is_no_label(unit), f"Error: {NO_LABEL} found in {self.__name__} in which it shouldn't exist!"


class WordVocab(BaseVocab):
    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False, ignore=[]):
        self.ignore = ignore
        super().__init__(data, lang=lang, idx=idx, cutoff=cutoff, lower=lower)
        self.state_attrs += ['ignore']

    def id2unit(self, id):
        if len(self.ignore) > 0 and id == EMPTY_ID:
            return '_'
        else:
            return super().id2unit(id)

    def unit2id(self, unit):
        assert not self.is_no_label(unit), f"Error: seeking {NO_LABEL} in {self.__name__} in which it shouldn't exist!"
        if len(self.ignore) > 0 and unit in self.ignore:
            return self._unit2id[EMPTY]
        else:
            return super().unit2id(unit)

    def build_vocab(self):
        if self.lower:
            counter = Counter([w[self.idx].lower() for sent in self.data for w in sent if w[self.idx].lower() != NO_LABEL.lower()])
        else:
            counter = Counter([w[self.idx] for sent in self.data for w in sent if w[self.idx] != NO_LABEL])
        for k in list(counter.keys()):
            if counter[k] < self.cutoff or k in self.ignore:
                del counter[k]

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

        for unit in self._id2unit: assert not self.is_no_label(unit), f"Error: {NO_LABEL} found in {self.__name__} in which it shouldn't exist!"

class XPOSVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="", keyed=False):
        raise NotImplementedError  # TODO NOLABEL/ignore if we're going to work on this.
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

class FeatureVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="|", keyed=True):
        raise NotImplementedError  # TODO NOLABEL/ignore if we're going to work on this.
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

class MultiVocab(BaseMultiVocab):
    def state_dict(self):
        """ Also save a vocab name to class name mapping in state dict. """
        state = OrderedDict()
        key2class = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
            key2class[k] = type(v).__name__
        state['_key2class'] = key2class
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        class_dict = {'CharVocab': CharVocab,
                'WordVocab': WordVocab}
        new = cls()
        assert '_key2class' in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop('_key2class')
        for k,v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new

