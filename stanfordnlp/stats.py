import os
import argparse
import pandas as pd
#from scipy.stats import permutation_test
from itertools import combinations


#def permutation_test(g1, g2, func):
#    stat = func(g1, g2)
#
#    union = g1 + g2
#
#    def aux(ia): # Set of indices
#        ga = []
#        gb = []
#        for i in range(len(union)):
#            if i in ia: ga.append(union[i])
#            else: gb.append(union[i])
#
#        return func(ga, gb)
#
#    l = [aux(ia) for ia in combinations(set(range(len(union))), len(g1))]
#
#    count = 0
#    for d in l:
#        if(d >= stat): count += 1
#
#    return stat, count / len(l)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


argparser = argparse.ArgumentParser()
argparser.add_argument("-ef", "--eval_files", required=True, help="separated with \";\"")
argparser.add_argument("-op", "--out_prefix", required=True, help="prefix of path to save csv")
argparser.add_argument("-gp", "--group_pairs", default=None, help="model group pairs to test; None (default) means all combinations")
argparser.add_argument("-k", "--topk", default=None, help="only test top k runs")
args = argparser.parse_args()

df_columns = ["model", "dataset", "training_method", "annotations", "eval_set", "run_id"]
perf_columns = [f"{m}_{s}" for m in ["las", "upos"] for s in ["f", ]]
df = pd.DataFrame(columns=df_columns)

for eval_file in args.eval_files.split(";"):
    if not eval_file: continue

    if "stanfordnlp" in eval_file:  # RMK ad-hoc
        model = "stanfordnlp"
    elif "mti" in eval_file:
        model = "mti"
    else:
        raise NotImplementedError

    with open(eval_file, "r") as ef:
        for line in ef.readlines():
            model_dir, ckpt_file, _, eval_set, _, _, _, _, _, lasa, lasb, lasc, uposa, uposb, uposc = line.split(" ")
            _, dataset, annotations, training_method = model_dir.split("_")
            if not training_method:
                if model == "stanfordnlp":
                    training_method = "MT"
                else:
                    raise NotImplementedError
            elif training_method == "BS":
                if model == "stanfordnlp":
                    training_method = "MTBS"
                else:
                    raise NotImplementedError
            elif training_method == "NT":
                if model == "stanfordnlp":
                    training_method = "STP"
                else:
                    raise NotImplementedError
            elif training_method == "NT":
                if model == "stanfordnlp":
                    training_method = "STT"
                else:
                    raise NotImplementedError
            assert not ckpt_file.split("_")[0][-2].isdigit()
            if model == "stanfordnlp":
                run_id = int(ckpt_file.split("_")[0][-1])  # Note that this is very ad-hoc!!!
            elif model == "mti":
                run_id = ckpt_file.split("_")[-1]
                if run_id.isdigit(): run_id = int(run_id)
                else: run_id = 0
            eval_set = eval_set[:-1]
            if eval_set.endswith("_DEV"): eval_set = eval_set[:-4]
            las = lasa + lasb + lasc
            upos = uposa + uposb + uposc
            las_p, las_r, las_f = eval(las.split("\t")[0])
            upos_p, upos_r, upos_f = eval(upos.strip())
            perfs = dict(las_p=las_p, las_r=las_r, las_f=las_f, upos_p=upos_p, upos_r=upos_r, upos_f=upos_f)
            df = df.append(dict(model=model, dataset=dataset, training_method=training_method, annotations=annotations, run_id=run_id, eval_set=eval_set, **perfs ), ignore_index=True)

df = df[df_columns + perf_columns]

groups = {name: group for name, group in df.groupby(df_columns[:5])}
if args.topk is not None:
    groups = {n: g.nlargest(int(args.topk), "las_f") for n, g in groups.items()}
    topk_name = f"_top{args.topk}"
else:
    topk_name = ""
if args.group_pairs is None:
    group_pairs = list(combinations(groups, 2))
else:
    exit()
    # TODO

means = pd.DataFrame(columns=[n for n in df_columns[0:5]] + ["performance", "value"])
for i, (n, g) in enumerate(groups.items()):
    print(f"1/2, processing {i+1}/{len(groups)} means")
    for pc in perf_columns:
        means = means.append(dict(model=n[0], dataset=n[1], training_method=n[2], annotations=n[3],
                                  performance=pc, eval_set=n[4], value=str(round(g[pc].mean(), 4)),
                                  ), ignore_index=True)
means.to_csv(os.path.join(args.out_prefix, f"means{topk_name}.csv"))
print()
exit()  # TODO comment

pitman_res = pd.DataFrame(columns=["g1_" + n for n in df_columns[1:4]] + ["g2_" + n for n in df_columns[1:4]] + ["performance", "value1", "value2", "mean_diff", "pvalue"])
for i, (n1, n2) in enumerate(group_pairs):
    print(f"2/2, processing {i+1}/{len(group_pairs)} tests")
    g1 = groups[n1]
    g2 = groups[n2]
    for pc in perf_columns:
        #test_res = permutation_test(g1[pc].to_list(), g2[pc].to_list(), lambda x, y: (sum(x) / len(x)) - (sum(y) / len(y)))
        test_res = permutation_test((g1[pc].to_numpy(), g2[pc].to_numpy()), lambda x, y: x.mean() - y.mean(), alternative="two-sided")
        pitman_res = pitman_res.append(dict(g1_dataset=n1[1], g1_training_method=n1[2], g1_annotations=n1[3],
                                            g2_dataset=n2[1], g2_training_method=n2[2], g2_annotations=n2[3],
                                            performance=pc, value1=g1[pc].mean(), value2=g2[pc].mean(), mean_diff=test_res.statistic, pvalue=test_res.pvalue,
                                            backward_copy="false",
                                            #alternative="two-sided",
                                            ), ignore_index=True)
        # The following is a copy to facilitate backward indexing when manually viewing the CSV file
        pitman_res = pitman_res.append(dict(g2_dataset=n1[1], g2_training_method=n1[2], g2_annotations=n1[3],
                                            g1_dataset=n2[1], g1_training_method=n2[2], g1_annotations=n2[3],
                                            performance=pc, value1=g2[pc].mean(), value2=g1[pc].mean(), mean_diff=-test_res.statistic, pvalue=test_res.pvalue,
                                            backward_copy="true",
                                            #alternative="two-sided",
                                            ), ignore_index=True)
pitman_res.to_csv(os.path.join(args.dir, f"pitman{topk_name}.csv"))
print()
