"""
Processor for performing dependency parsing
"""

from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.common.utils import unsort
from stanfordnlp.models.depparse.data import DataLoader
from stanfordnlp.models.depparse.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class POSDepparseMTProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([POSDEPPARSEMT])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # get pretrained word vectors
        self._pretrain = Pretrain(config['pretrain_path'])
        # set up trainer
        self._trainer = Trainer(pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, doc):
        batch = DataLoader(
            doc, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True, sort_saving_orig_idx=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        preds = unsort(preds, batch.data_unsorted_orig_idx)
        batch.conll.set(['head', 'deprel', 'upos'], [y for x in preds for y in x])
