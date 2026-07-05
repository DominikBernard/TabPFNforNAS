import argparse

from tabpfn_final_accuracy.utils.experiment_utils import run_experiment

parser = argparse.ArgumentParser(description="Run experiments to find the best count of epochs as features.")
parser.add_argument("--train_size", type=int, default=1000, help="Number of training samples.")
parser.add_argument("--seeds", type=int, nargs='+', default=[42, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300], help="List of random seeds.")
parser.add_argument("--skip_unconnected", default=False, action='store_true', help="Whether to skip unconnected architectures.")
parser.add_argument("--zero_cost_features", default=False, action='store_true', help="Whether to include zero-cost features.")
parser.add_argument("--epochs_n_features", type=int, nargs='+', default=[0, 1, 2, 5, 10, 15, 20, 25, 30, 40], help="List of counts of epochs to include as features.")


if __name__ == "__main__":
    args = parser.parse_args()

    zcf = "withZCF" if args.zero_cost_features else "withoutZCF"
    su = "skipUnconnected" if args.skip_unconnected else "NoSkipUnconnected"

    for seed in args.seeds:
        print(f"Running for seed {seed}...")
        for enf in args.epochs_n_features:
            logdir = f"tabpfn_final_accuracy/logs/epochs_n_features/{su}/{zcf}/seed_{seed}/{enf}_epochs"
            run_experiment(
                logdir=logdir,
                train_size=args.train_size,
                seed=seed,
                skip_unconnected=args.skip_unconnected,
                zero_cost_features=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'] if args.zero_cost_features else [],
                epochs_n_features=enf
            )
