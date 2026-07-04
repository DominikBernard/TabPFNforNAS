import itertools
import tqdm
import argparse

from tabpfn_with_nats_bench.utils.experiment_utils import run_experiment

parser = argparse.ArgumentParser(description="Run experiments with different combinations of zero-cost features.")
parser.add_argument("--train_size", type=int, default=1000, help="Number of training samples.")
parser.add_argument("--seeds", type=int, nargs='+', default=[42, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300], help="List of random seeds.")
parser.add_argument("--skip_unconnected", default=True, action='store_true', help="Whether to skip unconnected architectures.")
parser.add_argument("--zero_cost_features", type=str, nargs='+', default=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'], help="List of zero-cost features to consider.")

if __name__ == "__main__":
    args = parser.parse_args()

    feature_sets = []
    for r in range(len(args.zero_cost_features) + 1):
        for combo in itertools.combinations(args.zero_cost_features, r):
            feature_sets.append(list(combo))

    su_path = "skipUnconnected" if args.skip_unconnected else "NoSkipUnconnected"

    for seed in args.seeds:
        print(f"Running for seed {seed}...")
        for zcf in tqdm.tqdm(feature_sets):
            logdir = f"tabpfn_with_nats_bench/logs/zero_cost_combination/{su_path}/seed_{seed}/{'-'.join(zcf) if zcf else 'noZC'}"
            run_experiment(
                logdir=logdir,
                train_size=args.train_size,
                seed=seed,
                skip_unconnected=args.skip_unconnected,
                zero_cost_features=zcf
            )

