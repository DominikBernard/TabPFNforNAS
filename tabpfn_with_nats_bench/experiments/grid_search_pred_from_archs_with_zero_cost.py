# run multiple experiments with different hyperparameters
# using pred_from_archs_with_zero_cost_and_train_acc.py runned with different arguments

import itertools
import argparse

from tabpfn_with_nats_bench.utils.experiment_utils import run_experiment


parser = argparse.ArgumentParser(description="Run multiple experiments with different hyperparameters.")
parser.add_argument("--seeds", type=int, nargs='+', default=[42, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300], help="List of random seeds.")
parser.add_argument("--train_sizes", type=int, nargs='+', default=[10, 100, 1000, 10_000], help="List of training sizes.")
parser.add_argument("--connected_feature", type=bool, nargs='+', default=[True, False], help="Whether to include connected feature.")
parser.add_argument("--arch_index_feature", type=bool, nargs='+', default=[True, False], help="Whether to include architecture index feature.")
parser.add_argument("--edge_features", type=str, nargs='+', default=["none", "raw", "onehot"], help="List of edge feature types.")
parser.add_argument("--zero_cost_features", type=bool, nargs='+', default=[True, False], help="Whether to include all zero-cost features.")
parser.add_argument("--skip_unconnected", type=bool, nargs='+', default=[True, False], help="Whether to skip unconnected architectures.")
parser.add_argument("--epochs_n_features", type=int, nargs='+', default=[0, 10], help="List of counts of epochs to include as features.")

if __name__ == "__main__":
    args = parser.parse_args()

    # hyperparameter grids
    zero_cost_features = []
    if True in args.zero_cost_features:
        zero_cost_features.append(['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'])
    if False in args.zero_cost_features:
        zero_cost_features.append([])

    for s, ts, cf, aif, ef, zcf, su, enf in itertools.product(args.seeds, args.train_sizes, args.connected_feature, args.arch_index_feature, args.edge_features, zero_cost_features, args.skip_unconnected, args.epochs_n_features):
        # skip the combination where no features are included
        if cf == False and aif == False and ef == "none" and zcf == [] and enf == 0:
            continue

        zcf_bool = "withZC" if zcf else "noZC"
        su_bool = "skipUnconnected" if su else "noSkipUnconnected"

        logdir = f"tabpfn_with_nats_bench/logs/grid_search_results/seed_{s}/train-{ts}_cf-{cf}_aif-{aif}_ef-{ef}_{zcf_bool}_{su_bool}_enf-{enf}"

        run_experiment(
            logdir=logdir,
            train_size=ts,
            seed=s,
            connected_feature=cf,
            arch_index_feature=aif,
            edge_features=ef,
            zero_cost_features=zcf,
            skip_unconnected=su,
            epochs_n_features=enf
        )

