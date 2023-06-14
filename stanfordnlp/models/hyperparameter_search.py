import time
import torch.multiprocessing as mp
import numpy as np
from skopt import gp_minimize, Optimizer
from skopt.space import Real


def lr_search(f, args, lower, upper, prior="log-uniform", num_searches=20, xi=0.01):
    def callback(res):  # Callback called by skopt.gp_minimize for each point
        step_lr = res.x_iters[-1][0]
        step_perf = -res.func_vals[-1]
        current_best_lr = res.x[0]
        current_best_perf = -res.fun
        print(f"search step {len(res.x_iters)}/{num_searches}: lr={step_lr}\tperf={step_perf}")
        print(f"current best: lr={current_best_lr}\tperf={current_best_perf}")

    func = (lambda x: -f(x, args))  # Function minimised by skopt.gp_minimize.
    res = gp_minimize(func=func, dimensions=[Real(lower, upper, prior=prior), ], acq_func="gp_hedge", xi=xi, n_calls=num_searches, noise="gaussian", random_state=None, callback=callback)

    print("Search completed.")
    best_lr = res.x[0]
    best_perf = -res.fun
    print(f"\tBest lr: {best_lr}")
    print(f"\tBest performance: {best_perf}")
    print()

    all_lrs = map(lambda x: x[0], res.x_iters)
    all_perfs = map(lambda x: -x, res.func_vals)
    print("All searches:")
    for lr, perf in zip(all_lrs, all_perfs):
        print(f"lr={lr}\tperf={perf}")

    return best_lr, best_perf


#def lr_search(f, args, lower, upper, prior="log-uniform", parallel=4, num_searches=20):
#    optimizer = Optimizer(
#            dimensions=[Real(lower, upper, prior=prior), ],
#            random_state=1,
#            base_estimator='gp'
#            )
#
#    mp.set_start_method('spawn')
#
#    all_sampled_points = set()
#    all_evaluated_perfs = []
#    workers = set()
#    num_completed_searches = 0
#    q = mp.Queue()
#    while num_completed_searches < num_searches:
#        if len(workers) < parallel:
#            # Sample points for all idle workers
#            num_idle_workers = parallel - len(workers)
#            sampled_points = []
#            while len(sampled_points) < num_idle_workers:  # Continues until sampled_points contains #num_idle_workers unique values that have never been evaluated with before.
#                sampled_points += optimizer.ask(n_points=num_idle_workers, strategy="cl_max")  # x is a list of n_points points
#                sampled_points = list(set(sampled_point[0] for sampled_point in sampled_points).difference(all_sampled_points))  # removes duplicates
#                sampled_points = sampled_points[:parallel]
#            all_sampled_points = all_sampled_points.union(set(sampled_points))
#
#            # Distribute and evaluate the sampled points over the workers
#            for sampled_point in sampled_points:
#                worker = mp.Process(target=f, args=(sampled_point, args, q))
#                worker.start()
#                workers.add(worker)
#
#            # Check if there are any ready (idle) workers
#            for worker in workers.copy():
#                if not worker.is_alive():
#                    worker.join()
#                    workers.remove(worker)
#                    perfs, lr = q.get()
#                    all_evaluated_perfs.append(perfs)
#                    optimizer.tell([lr, ], perfs[0])  # optimizes using the main perf
#                    num_completed_searches += 1
#
#            if len(optimizer.yi) > 0:
#                max_i = np.argmax(optimizer.yi)
#                print("current best lr %s, main perf %s, perfs\n%s" % (optimizer.Xi[max_i], optimizer.yi[max_i], all_evaluated_perfs[max_i]))  # current best lr, main perf and perfs
#        else:
#            # Wait two seconds, so the while-condition isn't checked constantly
#            time.sleep(2)
#
#            # Check if there are any ready (idle) workers
#            for worker in workers.copy():
#                if not worker.is_alive():
#                    worker.join()
#                    workers.remove(worker)
#                    perfs, lr = q.get()
#                    all_evaluated_perfs.append(perfs)
#                    optimizer.tell([lr, ], perfs[0])  # optimizes using the main perf
#                    num_completed_searches += 1
#
#    max_i = np.argmax(optimizer.yi)
#    all_lrs = list(zip(optimizer.Xi, optimizer.yi, all_evaluated_perfs))
#    best_lr, best_main_perf, best_perfs = all_lrs[max_i]
#    print("Search ended.\nAll evaluated lrs\n%s\n\nBest lr %s, main perf %s, perfs\n%s" % (all_lrs, best_lr, best_main_perf, best_perfs))
#    return best_lr, best_main_perf, best_perfs
