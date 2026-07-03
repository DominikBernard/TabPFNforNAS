import argparse
import os
import datetime
import regex as re
import __main__
import matplotlib.pyplot as plt



def change_logdir_if_default(main_args: argparse.Namespace):
    """Change logdir if it is the default value."""
    # check if logdir is default
    if main_args.logdir == 'tabpfn_with_nats_bench/logs':
        # Create logdir name.
        main_args.logdir = os.path.join("tabpfn_with_nats_bench/logs", "{}-{}-{}".format(
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

def plot_one_scatter_plot(x, y, xlabel, ylabel, title, ax=None, output_file=None):
    """Plot a scatter plot of x vs y."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.plot([0, 100], [0, 100], 'r--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if output_file is not None:
        plt.savefig(output_file)


def plot_predictions_vs_true(test_labels, preds, logdir = None):

    output_file = os.path.join(logdir, "true_vs_pred.png") if logdir is not None else None
    # plot the results
    plot_one_scatter_plot(test_labels, preds, xlabel="True Labels", ylabel="Predictions", title="True Labels vs Predictions", output_file=output_file)
    
    # show the figure if not saved to file
    if logdir is None:
        plt.show()


def plot_two_predictions_vs_true(test_labels_1, test_labels_2, preds1, preds2, title1="Predikce 1", title2="Predikce 2", path_to_save=None):
    # plot the results next to each other
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    plot_one_scatter_plot(test_labels_1, preds1, xlabel="Skutečné hodnoty", ylabel="Predikce", title=title1, ax=axs[0])

    plot_one_scatter_plot(test_labels_2, preds2, xlabel="Skutečné hodnoty", ylabel="Predikce", title=title2, ax=axs[1])
    axs[1].set_ylabel("")

    # set number of ticks on x and y axis to 10
    axs[0].set_xticks(range(0, 101, 10))
    axs[0].set_yticks(range(0, 101, 10))
    axs[1].set_xticks(range(0, 101, 10))
    axs[1].set_yticks(range(0, 101, 10))

    plt.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.show()




