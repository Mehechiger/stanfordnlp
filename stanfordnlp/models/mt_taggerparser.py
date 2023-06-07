"""
Entry point for training and evaluating a multitask pos tagger + dependency parser.
"""

"""
Training and evaluation for the mt_taggerparser.
"""

import sys
import os
import shutil
import logging
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch
from torch import nn, optim

from stanfordnlp.models.posdepparse_mt.data import DataLoader
from stanfordnlp.models.posdepparse_mt.trainer import Trainer
from stanfordnlp.models.posdepparse_mt import scorer
from stanfordnlp.models.common import utils
from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.hyperparameter_search import lr_search


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/depparse', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--glove_dir', type=str, default='extern_data/glove', help='Directory of GloVe vectors')
    parser.add_argument('--pretrained_vec', type=str, default='glove', help='word2vec or glove')
    parser.add_argument('--glove_B', type=str, default=6, help='GloVe pretraining number of words')
    parser.add_argument('--glove_dim', type=str, default=100, help='GloVe vectors dim')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Gold CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--fix_emb_dim', type=int, default=32)  # fix_embedding_size=8 in MTI  # RMK 32 seems better for this model
    parser.add_argument('--char_type', default="fix", help="char(actor embeddings), (pre/suf)fix (embeddings) or deactivated")
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=2)  # default ndf=2 in MTI; default in stanfordnlp: 3 from parser (2 for tagger)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    parser.add_argument('--pretrain_restrict_to_train_vocab', action='store_true')  # whether filtering out vocabs not seen in train, like in MTI  # RMK makes no difference in practice
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false', help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--fix_eval_interval', dest='adapt_eval_interval', action='store_false', help="Use fixed evaluation interval for all treebanks, otherwise by default the interval will be increased for larger treebanks.")
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/posdepparsemt', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--search_lr', action='store_true', help='Searches (bayesian) best lr (ignores --lr; will be ignored if --mode is predict).')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    print("Running tagger-parser in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        if args["search_lr"]:
            return search_lr(args)
        else:
            return train(args)
    else:
        return evaluate(args)


def _search_lr_aux_train_func(lr, args, q):
    args["lr"] = lr
    now = datetime.now().strftime("%y%m%d%H%M%S%f")
    args["save_name"] = f"lr_search_{now}"
    args["output_file"] = f"lr_search_{now}.conllu"
    res = train(args)
    q.put(((res[0][0], res), lr))


def search_lr(args):
    return lr_search(_search_lr_aux_train_func, args, 0.00003, 3, parallel=4, num_searches=20)


def train(args):
    utils.ensure_dir(args['save_dir'])
    model_file = '{}/{}_{}_mt_taggerparser.pt'.format(args['save_dir'], args['save_name'], args['shorthand']) if args['save_name'] is not None else '{}/{}_mt_taggerparser.pt'.format(args['save_dir'], args['shorthand'])

    # Handles logging
    train_log_file = model_file + ".train.log"
    logging.basicConfig(level=logging.DEBUG)
    train_logger = logging.getLogger()
    train_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(levelname)s - {args["save_name"]} - %(message)s')
    file_handler = logging.FileHandler(train_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    train_logger.handlers = []
    train_logger.addHandler(file_handler)
    train_logger.addHandler(stream_handler)

    # load pretrained vectors  # TODO 1\* use the same prept
    if args["pretrained_vec"] == "word2vec":
        vec_file = utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    elif args["pretrained_vec"] == "glove":
        assert args["shorthand"].split("_")[0] == "en"  # TODO currently the only supported language.
        vec_file = utils.get_glove_file(args['glove_dir'], args['glove_B'], args['glove_dim'])
    else:
        raise NotImplementedError
    pretrain_file = f"{args['save_dir']}/glove.{args['glove_B']}B.{args['glove_dim']}d.{args['train_file'].split('/')[-1]}.pretrain.pt"
    pretrain = Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])

    # load data
    train_logger.info("Loading data with batch size {}...".format(args['batch_size']))
    train_batch = DataLoader(args['train_file'], args['batch_size'], args, pretrain, evaluation=False)
    vocab = train_batch.vocab
    dev_batch = DataLoader(args['eval_file'], args['batch_size'], args, pretrain, vocab=vocab, evaluation=True, pretrain_restrict_to_train_vocab=args['pretrain_restrict_to_train_vocab'])

    # pred and gold path
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        train_logger.error("Skip training because no data available...")
        sys.exit(0)

    train_logger.info("Training mt_taggerparser...")
    trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])

    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    if args['adapt_eval_interval']:
        args['eval_interval'] = utils.get_adaptive_eval_interval(dev_batch.num_examples, 2000, args['eval_interval'])
        train_logger.info("Evaluating the model every {} steps...".format(args['eval_interval']))

    using_amsgrad = False
    last_best_step = 0
    # start training
    train_loss = 0
    while True:
        do_break = False
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False) # update step
            train_loss += loss
            if global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                train_logger.info(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step, max_steps, loss, duration, current_lr))

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                train_logger.info("Evaluating on dev set...")
                dev_preds = []
                for batch in dev_batch:
                    preds = trainer.predict(batch)
                    dev_preds += preds

                dev_batch.combined.set_conll(['head', 'deprel', 'upos'], [y for x in dev_preds for y in x])
                dev_batch.combined.write_conll(system_pred_file)
                _, _, dev_score_parser, _, _, dev_score_tagger = scorer.score(system_pred_file, gold_file)

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                train_logger.info("step {}: train_loss = {:.6f}, dev_score_parser = {:.4f}, dev_score_tagger = {:.4f}".format(global_step, train_loss, dev_score_parser, dev_score_tagger))
                train_loss = 0

                # save best model # RMK based on parser score
                if len(dev_score_history) == 0 or dev_score_parser > max(list(zip(*dev_score_history))[0]):
                    last_best_step = global_step
                    trainer.save(model_file)
                    train_logger.info("new best model saved.")
                    best_dev_preds = dev_preds

                dev_score_history += [(dev_score_parser, dev_score_tagger)]
                train_logger.info("")

            if global_step - last_best_step >= args['max_steps_before_stop']:
                if not using_amsgrad:
                    train_logger.info("Switching to AMSGrad")
                    last_best_step = global_step
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']), eps=1e-6)
                else:
                    do_break = True
                    break

            if global_step >= args['max_steps']:
                do_break = True
                break

        if do_break: break

        train_batch.reshuffle()

    best_eval_parser_ind, best_eval_tagger_ind = np.argmax(list(zip(*dev_score_history)), axis=1)
    best_eval_parser = best_eval_parser_ind + 1
    best_eval_tagger = best_eval_tagger_ind + 1
    best_f_parser, best_f_parser_on_tagging = dev_score_history[best_eval_parser_ind]
    best_f_tagger_on_parsing, best_f_tagger = dev_score_history[best_eval_tagger_ind]
    train_logger.info("Training ended with {} steps.".format(global_step))
    train_logger.info("Best dev F1 parser = {:.4f}, at iteration = {}, on tagging = {:.4f}".format(best_f_parser, best_eval_parser * args['eval_interval'], best_f_parser_on_tagging))
    train_logger.info("Best dev F1 tagger = {:.4f}, at iteration = {}, on parsing = {:.4f}".format(best_f_tagger, best_eval_tagger * args['eval_interval'], best_f_tagger_on_parsing))

    return (best_f_parser, best_f_parser_on_tagging, best_eval_parser), (best_f_tagger, best_f_tagger_on_parsing, best_eval_tagger)


def evaluate(args):
    # file paths
    system_pred_file = args['output_file']
    gold_file = args['gold_file']
    model_file = args['save_dir'] + '/' + args['save_name']
    pretrain_file = f"{args['save_dir']}/glove.{args['glove_B']}B.{args['glove_dim']}d.{args['train_file'].split('/')[-1]}.pretrain.pt"

    # Handles logging
    eval_log_file = model_file + ".eval.log"
    logging.basicConfig(level=logging.DEBUG)
    eval_logger = logging.getLogger()
    eval_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler = logging.FileHandler(eval_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    eval_logger.handlers = []
    eval_logger.addHandler(file_handler)
    eval_logger.addHandler(stream_handler)

    # load pretrain
    pretrain = Pretrain(pretrain_file)

    # load model
    eval_logger.info("Loading model from: {}".format(model_file))
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(pretrain=pretrain, model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
            loaded_args[k] = args[k]

    # load data
    eval_logger.info("Loading data with batch size {}...".format(args['batch_size']))
    batch = DataLoader(args['eval_file'], args['batch_size'], loaded_args, pretrain, vocab=vocab, evaluation=True, pretrain_restrict_to_train_vocab=args['pretrain_restrict_to_train_vocab'])

    if len(batch) > 0:
        eval_logger.info("Start evaluation...")
        preds = []
        for i, b in enumerate(batch):
            preds += trainer.predict(b)
    else:
        # skip eval if dev data does not exist
        preds = []

    # write to file and score
    batch.combined.set_conll(['head', 'deprel', 'upos'], [y for x in preds for y in x])
    batch.combined.write_conll(system_pred_file)

    if gold_file is not None:
        _, _, score_parser, _, _, score_tagger = scorer.score(system_pred_file, gold_file)

        eval_logger.info(f"MT_TaggerParser {args['save_name']} score:")
        eval_logger.info("parser {} {:.4f}".format(args['shorthand'], score_parser))
        eval_logger.info("tagger {} {:.4f}".format(args['shorthand'], score_tagger))
    else:
        score_parser = None
        score_tagger = None

    return (score_parser, score_tagger), preds


if __name__ == '__main__':
    main()
