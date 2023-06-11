import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from stanfordnlp.models.common.combined import NO_LABEL
from stanfordnlp.models.common.hlstm import HighwayLSTM
from stanfordnlp.models.common.dropout import WordDropout
from stanfordnlp.models.common.char_model import CharacterModel

class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['char_type'] == "char" and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        elif self.args['char_type'] == "fix" and self.args['fix_emb_dim'] > 0:
            self.prefix_emb = nn.Embedding(len(vocab['prefix']), self.args['fix_emb_dim'], padding_idx=0)
            self.suffix_emb = nn.Embedding(len(vocab['suffix']), self.args['fix_emb_dim'], padding_idx=0)
            input_size += self.args['fix_emb_dim'] * 2
        elif self.args['char_type'] == 'deactivated':
            pass
        else:
            raise NotImplementedError

        # RMK: https://stanfordnlp.github.io/stanfordnlp/training.html#preparing-word-vector-data
        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        
        # recurrent layers
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0) # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, pretrained, word_orig_idx, sentlens, wordlens):
        if not (self.training and (upos == NO_LABEL)):  # Saves time when we do not need to calculate the loss nor the preds (i.e. training but with no available supervision).
            def pack(x):
                return pack_padded_sequence(x, sentlens, batch_first=True)

            inputs = []
            if self.args['pretrain']:
                pretrained_emb = self.pretrained_emb(pretrained)
                pretrained_emb = self.trans_pretrained(pretrained_emb)
                pretrained_emb = pack(pretrained_emb)
                inputs += [pretrained_emb]

            def pad(x):
                return pad_packed_sequence(PackedSequence(x, pretrained_emb.batch_sizes), batch_first=True)[0]

            if self.args['char_type'] == "char" and self.args['char_emb_dim'] > 0:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
                inputs += [char_reps]
            elif self.args['char_type'] == "fix" and self.args['fix_emb_dim'] > 0:
                prefixes, suffixes = wordchars
                prefix_reps = self.prefix_emb(prefixes).sum(dim=2)
                prefix_reps = pack(self.drop(prefix_reps))
                inputs += [prefix_reps]
                suffix_reps = self.suffix_emb(suffixes).sum(dim=2)
                suffix_reps = pack(self.drop(suffix_reps))
                inputs += [suffix_reps]
            elif self.args['char_type'] == 'deactivated':
                pass
            else:
                raise NotImplementedError

            lstm_inputs = torch.cat([x.data for x in inputs], 1)
            lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
            lstm_inputs = self.drop(lstm_inputs)
            lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

            lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
            lstm_outputs = lstm_outputs.data

            upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
            upos_pred = self.upos_clf(self.drop(upos_hid))

        if self.training:
            preds = []
            if upos == NO_LABEL:  # No available tagging supervision.
                loss = 0
            else:
                upos = pack(upos).data
                loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))
        else:
            loss = 0
            preds = [pad(upos_pred).max(2)[1]]

        return loss, preds
