# RMK Be careful this script is highly ad-hoc.
import os
import random
import numpy as np
import torch
from stanfordnlp.models.mt_taggerparser import get_arg_parser, evaluate
from stanfordnlp.utils.conll18_ud_eval import load_conllu_file
from stanfordnlp.utils.conll18_ud_eval import evaluate as ud_eval

arg_parser = get_arg_parser()
arg_parser.add_argument('-ed', '--exp_dir', type=str, required=True)
arg_parser.add_argument('-olp', '--out_log_path', type=str)
arg_parser.add_argument('-olm', '--out_log_mode', type=str, required=True, choices=["a", "w", "off"], help="off for no ud eval thus no out log (pred only).")
arg_parser.add_argument('-oof', '--overwrite_out_files', action='store_true')
arg_parser.add_argument("-itpo", "--include_these_prefixes_only", type=str, help="separated with ';'")
args = arg_parser.parse_args()

assert not (args.no_tagging and args.no_parsing)

args.mode = "predict"
args.save_dir = None
args.save_name = None
args.search_lr = False
assert args.sample_train == 1.0


endswith = "_mt_taggerparser.pt"


if args.out_log_path is None: args.out_log_path = os.path.join(args.exp_dir, "eval.log")

if args.out_log_mode != "off": fol = open(args.out_log_path, args.out_log_mode)

if args.include_these_prefixes_only is not None:
    prefixes = args.include_these_prefixes_only.split(";")
else:
    prefixes = ["", ]  # skips nothing

for pt_file in os.listdir(args.exp_dir):
    if not pt_file.endswith(endswith): continue

    skip = True
    for prefix in prefixes:
        if pt_file.startswith(prefix):
            skip = False
            break
    if skip: continue

    pt_long_name = pt_file[:-len(endswith)]  # removes "endswith"
    pt_short_name = pt_long_name.split("_")[0]

    if "compl" in pt_short_name:
        annot_type = "COMPL"
    elif "disj" in pt_short_name:
        annot_type = "DISJ"
    else:
        raise NotImplementedError(f"Unknown pt_short_name {pt_short_name} of pt_full_name {pt_long_name} of pt_file {pt_file}")

    if "dm" in pt_short_name:
        dataset = "PTB"
        dst = "ptb"
        annot_id = pt_long_name.split("_")[1]
    elif "ud" in pt_short_name:
        dataset = "UD"
        dst = "gum+ewt"
        annot_id = "_".join(pt_long_name.split("_")[1:3])
    elif "gum" in pt_short_name:
        dataset = "GUM"
        dst = "gum"
        annot_id = pt_long_name.split("_")[1]
    elif "ewt" in pt_short_name:
        dataset = "EWT"
        dst = "ewt"
        annot_id = pt_long_name.split("_")[1]
    else:
        raise NotImplementedError(f"Unknown pt_short_name {pt_short_name} of pt_full_name {pt_long_name} of pt_file {pt_file}")

    args.shorthand = args.lang + f"_{dst}"
    args.train_file = f"en_{dst}.train_{annot_id}.in.combined"

    if "bs" in pt_short_name:
        training = "BS"
        args.training_label_complementing_strategy = "bootstrap"
    elif "nt" in pt_short_name:
        training = "NT"
        args.no_tagging = True
    elif "np" in pt_short_name:
        training = "NP"
        args.no_parsing = True
    else:
        training = ""
        args.training_label_complementing_strategy = None

    if dataset == "PTB":
        test_files = {"PTB": (os.path.join(args.data_dir, "en_ptb.dev.in.combined"), os.path.join(args.data_dir, "en_ptb.dev.gold.conllu")), }
    elif dataset in ["UD", "GUM", "EWT"]:
        test_files = {}
        for test_set_name, tmp in [("UD", "gum+ewt"), ("GUM_DEV", "gum"), ("EWT_DEV", "ewt")]: test_files[test_set_name] = (os.path.join(args.data_dir, f"en_{tmp}.dev.in.combined"), os.path.join(args.data_dir, f"en_{tmp}.dev.gold.conllu"))
    else:
        raise NotImplementedError

    model_dir__ = f"_{dataset}_{annot_type}_{training}"

    with open(os.path.join(args.exp_dir, f"{pt_long_name}_mt_taggerparser.pt.train.log"), "r") as logf:
        lines = logf.readlines()
        if "Training ended with" not in lines[-3]:
            print(f"Training not finished yet, skipping {model_dir__} {pt_file}")
            continue

    for test_set_name, (dev_file, gold_file) in test_files.items():
        if test_set_name.startswith(dataset):
            out_file = os.path.join(args.exp_dir, f"{pt_long_name}.dev.pred")
        else:
            out_file = os.path.join(args.exp_dir, f"{pt_long_name}.evalOn_{test_set_name}.pred")
        if os.path.exists(out_file + ".conllu") and (not args.overwrite_out_files):  # Skips to evaluation for those already evaluated.
            print(f"Already pred, skipping {model_dir__} {pt_file} evalOn {test_set_name}")
        else:
            if os.path.exists(out_file):
                if os.path.isfile(out_file):
                    os.remove(out_file)  # Removes the (presumably) unfinished tmp file.
                else:
                    raise NotImplementedError  # Some thing went wrong.

            args.save_dir = args.exp_dir
            args.save_name = pt_file
            args.gold_file = gold_file
            args.eval_file = dev_file
            args.output_file = out_file

            if args.seed is not None:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                random.seed(args.seed)
            if args.cpu:
                args.cuda = False
            elif args.cuda and args.seed is not None:
                torch.cuda.manual_seed(args.seed)

            args_dict = vars(args)
            print("Running tagger-parser in {} mode".format(args_dict['mode']))

            (las_p, las_r, las_f, upos_p, upos_r, upos_f), _ = evaluate(args_dict)

            os.rename(out_file, out_file + ".conllu")  # Renames the tmp file to final name.

        out_file += ".conllu"

        if args.out_log_mode != "off":
            gold_ud = load_conllu_file(gold_file)
            system_ud = load_conllu_file(out_file)
            evaluation = ud_eval(gold_ud, system_ud)
            las, upos = evaluation['LAS'], evaluation['UPOS']
            lass = las.precision, las.recall, las.f1
            uposs = upos.precision, upos.recall, upos.f1
            tmp = f'{model_dir__} {pt_file} evalOn {test_set_name}: UD eval (prec, rec, f1)\tLAS: {lass}\tUPOS: {uposs}\n'
            fol.write(tmp)
            print(tmp)

if args.out_log_mode != "off":
    fol.close()
