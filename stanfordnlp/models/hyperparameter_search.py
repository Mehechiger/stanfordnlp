import numpy as np
import os
from skopt import gp_minimize  #, Optimizer
from skopt.space import Real

from stanfordnlp.models.common import utils


def read_search_log(search_log, read_only=False):
    x0, y0 = [], []

    if os.path.isfile(search_log):
        logf = open(search_log, 'r')
        for line in logf.readlines():
            if line.startswith("#"):
                lr, perf = line[1:].split("\t")
                x0.append([eval(lr[3:]), ])
                y0.append(-eval(perf[5:]))
        if not read_only:
            print(f"loaded {len(x0)} points:")
            for x0_, y0_ in zip(x0, y0):
                print(f"lr={x0_}\tperf={-y0_}")
        logf.close()
        if not read_only:
            logf = open(search_log, 'a')
    elif not read_only:
        logf = open(search_log, 'w')
    else:
        logf = None

    if (len(x0) == 0):
        x0, y0 = None, None

    return x0, y0, logf


def lr_search(f, args, lower, upper, prior="log-uniform", num_searches=20, n_initial_points=10, xi=0.01):
    utils.ensure_dir(args['save_dir'])
    assert args['save_name'] is not None
    search_log = '{}/{}_{}_mt_taggerparser.search.txt'.format(args['save_dir'], args['save_name'], args['shorthand'])

    x0, y0, logf = read_search_log(search_log)
    len_x0 = 0 if x0 is None else len(x0)

    for n_calls in range(1, num_searches - len_x0 + 1):
        def callback(res):  # Callback called by skopt.gp_minimize for each point
            step_lr = res.x_iters[-1][0]
            step_perf = -res.func_vals[-1]
            print(f"search step {len(res.x_iters)}/{num_searches}: lr={step_lr}\tperf={step_perf}")

            if len(res.x_iters) > len_x0:
                logf.write(f"#lr={step_lr}\tperf={step_perf}\n")
                logf.flush()

            current_best_lr = res.x[0]
            current_best_perf = -res.fun
            print(f"current best: lr={current_best_lr}\tperf={current_best_perf}")

        # RMK due to the skopt implementation, the actual used n_initial_points = n_initial_points + len(x0)
        if len_x0 < 10:
            n_initial_points = 1 - len_x0
        else:
            n_initial_points = -len_x0

        func = (lambda x: -f(x, args))  # Function minimised by skopt.gp_minimize.
        res = gp_minimize(func=func, dimensions=[Real(lower, upper, prior=prior), ], acq_func="gp_hedge", xi=xi, n_calls=n_calls, noise="gaussian", random_state=None, callback=callback, x0=x0, y0=y0, n_initial_points=n_initial_points)

        logf.close()
        x0, y0, logf = read_search_log(search_log)
        len_x0 = 0 if x0 is None else len(x0)
    logf.close()

    print("Search completed.")
    ind = np.argmax(y0)
    best_lr = x0[ind][0]
    best_perf = -y0[ind]
    print(f"\tBest lr: {best_lr}")
    print(f"\tBest performance: {best_perf}")
    print()

    all_lrs = map(lambda x: x[0], x0)
    all_perfs = map(lambda x: -x, y0)
    print("All searches:")
    for lr, perf in zip(all_lrs, all_perfs):
        print(f"lr={lr}\tperf={perf}")

    logf = open(search_log, 'r')
    lines = logf.readlines()
    logf.close()
    if not (lines[-2].startswith("\tBest lr: ") and lines[-1].startswith("\tBest performance: ")):
        logf = open(search_log, 'a')
        logf.write(f"\tBest lr: {best_lr}\n")
        logf.write(f"\tBest performance: {best_perf}\n")
        logf.flush()
        logf.close()

    return best_lr, best_perf, all_lrs, all_perfs
