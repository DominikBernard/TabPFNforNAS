import argparse
import os
import datetime
import regex as re
import __main__
import matplotlib.pyplot as plt



def change_logdir_if_default(main_args: argparse.Namespace):
    """Change logdir if it is the default value."""
    # check if logdir is default
    if main_args.logdir == 'logs':
        # Create logdir name.
        main_args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(__main__.__file__),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(main_args).items())))
        ))
    os.makedirs(main_args.logdir, exist_ok=True)


def verbose_meta_info(args: argparse.Namespace):
    """Print verbose meta information about the experiment."""
    if args.verbose:
        print("Experiment Meta Information:")
        for arg, value in vars(args).items():
            if arg != "logdir":
                if arg == "zero_cost_features":
                    if value != []:
                        print("  included zero-cost features:")
                        if "f" in value:
                            print("    - f: Fisher Information")
                        if "g_n" in value:
                            print("    - g_n: Gradient Norm")
                        if "g" in value:
                            print("    - g: Grasp")
                        if "j_c" in value:
                            print("    - j_c: Jacobian Covariance")
                        if "p" in value:
                            print("    - p: Plain")
                        if "s" in value:
                            print("    - s: SNIP")
                        if "sy" in value:
                            print("    - sy: Synflow (log scaled)")
                        if "sy_bn" in value:
                            print("    - sy_bn: Synflow BN")
                    else:
                        print("  included zero-cost features: None")
                else:
                    print(f"  {arg}: {value}")
        print()


def plot_predictions_vs_true(test_labels, preds, logdir = None):
    # plot the results
    plt.figure(figsize=(7, 7))
    plt.scatter(test_labels, preds, alpha=0.5)

    # set x and y limits to 0-100
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    # add the y=x line
    plt.plot([0, 100], [0, 100], 'r--')

    plt.xlabel("True Labels")
    plt.ylabel("Predictions")
    plt.title("True Labels vs Predictions")
    
    # save the figure or show it
    if logdir is not None:
        plt.savefig(os.path.join(logdir, "true_vs_pred.png"))
    else:
        plt.show()
