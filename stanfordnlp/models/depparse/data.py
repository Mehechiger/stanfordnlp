import random
import logging
import torch

from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import combined
from stanfordnlp.models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID
from stanfordnlp.models.common.combined import NO_LABEL
from stanfordnlp.models.common.utils import get_prefixes, get_suffixes, unsort
from stanfordnlp.models.pos.vocab import CharVocab, PrefixVocab, SuffixVocab, WordVocab, MultiVocab
from stanfordnlp.pipeline.doc import Document


def check_annots(data):
    for sent in data:  # 1(ptbpos) or 2(ptbhead)+3(ptbdeprel) could be NO_LABEL.
        has_tag = True
        has_syn = True
        if sent[0][1] == NO_LABEL:
            for w in sent:
                assert w[1] == NO_LABEL and w[2] != NO_LABEL and w[3] != NO_LABEL
            has_tag = False
        elif sent[0][2] == NO_LABEL or sent[0][3] == NO_LABEL:
            for w in sent:
                assert w[1] != NO_LABEL and w[2] == NO_LABEL and w[3] == NO_LABEL
            has_syn = False
        sent += [has_tag, has_syn]
    return data


class DataLoader:
    def __init__(self, input_src, batch_size, args, pretrain, vocab=None, evaluation=False, sort_saving_orig_idx=False, pretrain_restrict_to_train_vocab=False):
        self.batch_size = batch_size
        self.args = args
        self.do_tagging = not args["no_tagging"]
        self.do_parsing = not args["no_parsing"]
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_saving_orig_idx = sort_saving_orig_idx
        self.pretrain_restrict_to_train_vocab = pretrain_restrict_to_train_vocab
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith('combined'), "Loaded file must be combined file."
            self.combined, data = self.load_file(filename, evaluation=self.eval)  # sent in data: ['form', 'ptbpos', 'ptbhead', 'ptbdeprel'], ..., has_tag, has_syn
        elif isinstance(input_src, Document):
            raise NotImplementedError  #  TODO currently not supported.
            filename = None
            doc = input_src
            self.conll, data = self.load_doc(doc)

        self._init_aux_(args, pretrain, vocab, data)

        if filename is not None:
            self.logger.info("{} batches created for {}.".format(len(self.data), filename))

    def init_with_complemented(self):
        self._init_aux_(None, None, None, self.reload_complemented_from_combined())

    def _init_aux_(self, args, pretrain, vocab, data):
        self.shuffled = not self.eval

        if args is None:
            assert pretrain is None
            assert vocab is None
        else:
            # handle vocab
            if vocab is None:
                self.vocab = self.init_vocab([sent[:-2] for sent in data])  # sent: ['form', 'ptbpos', 'ptbhead', 'ptbdeprel'], ..., has_tag, has_syn
            else:
                self.vocab = vocab
            self.pretrain_vocab = pretrain.vocab

            # filter and sample data
            if args.get('sample_train', 1.0) < 1.0 and not self.eval:
                keep = int(args['sample_train'] * len(data))
                data = random.sample(data, keep)
                self.logger.info("Subsample training set with rate {:g}".format(args['sample_train']))

        # before: data: ['form', 'ptbpos', 'ptbhead', 'ptbdeprel'], ..., has_tag, has_syn
        data = self.preprocess(data, self.vocab, self.pretrain_vocab)
        # shuffle for training
        if self.shuffled:
            tmp = list(enumerate(data))
            random.shuffle(tmp)
            self.data_unshuffled_orig_idx, data = zip(*tmp)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)

    # sent in data: ['form', 'ptbpos', 'ptbhead', 'ptbdeprel']
    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist

        multivocab_dict = {}
        if self.args["char_type"] == "char":
            charvocab = CharVocab(data, self.args['shorthand'])
            multivocab_dict["char"] = charvocab
        elif self.args["char_type"] == "fix":
            prefixvocab = PrefixVocab(data, self.args['shorthand'])
            suffixvocab = SuffixVocab(data, self.args['shorthand'])
            multivocab_dict["prefix"] = prefixvocab
            multivocab_dict["suffix"] = suffixvocab
        elif self.args["char_type"] == "deactivated":
            pass
        else:
            raise NotImplementedError

        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        multivocab_dict["word"] = wordvocab

        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        multivocab_dict["upos"] = uposvocab

        deprelvocab = WordVocab(data, self.args['shorthand'], idx=3)
        multivocab_dict["deprel"] = deprelvocab

        vocab = MultiVocab(multivocab_dict)
        return vocab

    # sent in data: ['form', 'ptbpos', 'ptbhead', 'ptbdeprel'], ..., has_tag, has_syn
    # Return: [word, char/(prefix, suffix), upos, pretrained, head, deprel, has_tag, has_syn]
    def preprocess(self, data, vocab, pretrain_vocab):
        processed = []
        for sent in data:
            has_tag, has_syn = sent[-2:]
            sent = sent[:-2]
            processed_sent = [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]  # form
            if self.args["char_type"] == "char":
                processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]  # form
            elif self.args["char_type"] == "fix":
                prefixes = [[ROOT_ID]] + [vocab['prefix'].map([prefixes for prefixes in get_prefixes(w[0], 3)]) for w in sent]
                suffixes = [[ROOT_ID]] + [vocab['suffix'].map([suffixes for suffixes in get_suffixes(w[0], 3)]) for w in sent]
                processed_sent += [(prefixes, suffixes)]  # form
            elif self.args["char_type"] == "deactivated":
                processed_sent += [[None, ] * (len(sent) + 1), ]
            else:
                raise NotImplementedError
            if has_tag and self.do_tagging:
                processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]  # ptbpos
            else:
                processed_sent += [[NO_LABEL, ] * (len(sent) + 1), ]
            processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0] for w in sent], vocab if self.pretrain_restrict_to_train_vocab else None)]  # form
            if has_syn and self.do_parsing:
                processed_sent += [[to_int(w[2], ignore_error=self.eval) for w in sent]]  # ptbhead
                processed_sent += [vocab['deprel'].map([w[3] for w in sent])]  # ptbdeprel
            else:
                processed_sent += [[NO_LABEL, ] * len(sent), [NO_LABEL, ] * len(sent)]
            processed_sent += [has_tag, has_syn]
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]  # [word, char/(prefix, suffix), upos, pretrained, head, deprel, has_tag, has_syn]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 8, str(batch)

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        if self.args["char_type"] == "char":
            wordchars = get_long_tensor(batch_words, len(word_lens))
            wordchars_mask = torch.eq(wordchars, PAD_ID)
        elif self.args["char_type"] == "fix":
            prefixes, suffixes = zip(*batch[1])
            wordchars = [get_long_tensor(prefixes, len(lens)), get_long_tensor(suffixes, len(lens))]
            wordchars_mask = None  # not used
        elif self.args["char_type"] == "deactivated":
            wordchars = None
            wordchars_mask = None
        else:
            raise NotImplementedError
        upos = get_long_tensor(batch[2], batch_size)
        pretrained = get_long_tensor(batch[3], batch_size)
        sentlens = [len(x) for x in batch[0]]
        head = get_long_tensor(batch[4], batch_size)
        deprel = get_long_tensor(batch[5], batch_size)
        has_tag = get_long_tensor(batch[6], batch_size)
        has_syn = get_long_tensor(batch[7], batch_size)
        return words, words_mask, wordchars, wordchars_mask, upos, pretrained, head, deprel, orig_idx, word_orig_idx, sentlens, word_lens, has_tag, has_syn

    # data: ['form', 'ptbpos', 'ptbhead', 'ptbdeprel'], ..., has_tag, has_syn
    # evaluation: if True, does not load gold annotations
    def load_file(self, filename, evaluation=False):
        # RMK: this is necessary since in their original design there are no gold annotations in the dev.in.conllu used for evaluation,
        # but in our combined files gold annotations are always there.
        if evaluation: excluded = ['ptbpos', 'ptbhead', 'ptbdeprel']
        else: excluded = []
        combined_file = combined.CombinedFile(filename, excluded=excluded)
        data = combined_file.get(['form', 'ptbpos', 'ptbhead', 'ptbdeprel'], as_sentences=True)
        data = check_annots(data)
        return combined_file, data

    def reload_complemented_from_combined(self):
        data = self.combined.get(['form', 'ptbpos', 'ptbhead', 'ptbdeprel'], as_sentences=True, complemented=True)
        data = check_annots(data)
        return data

    def load_doc(self, doc):
        raise NotImplementedError  # TODO
        data = doc.combined_file.get(['form', 'upos', 'head', 'deprel'], as_sentences=True)
        return doc.combined_file, data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]

        # Restores to original order before reshuffling, otherwise we will loose track of the original indices.
        if self.sort_saving_orig_idx:
            data = unsort(data, self.data_unsorted_orig_idx)
        if self.shuffled:
            data = unsort(data, self.data_unshuffled_orig_idx)

        self.data = self.chunk_batches(data)

        tmp = list(enumerate(data))
        random.shuffle(tmp)
        self.data_unshuffled_orig_idx, data = zip(*tmp)

    def chunk_batches(self, data):
        res = []

        if self.sort_saving_orig_idx:
            (data, ), self.data_unsorted_orig_idx = sort_all([data], [len(x[0]) for x in data])
        elif not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key = lambda x: len(x[0]), reverse=random.random() > .5)

        current = []
        currentlen = 0
        for x in data:
            if len(x[0]) + currentlen > self.batch_size:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

        if currentlen > 0:
            res.append(current)

        return res

def to_int(string, ignore_error=False):
    try:
        res = int(string)
    except ValueError as err:
        if ignore_error:
            return 0
        else:
            raise err
    return res
