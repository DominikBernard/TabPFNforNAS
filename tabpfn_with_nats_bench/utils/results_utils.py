import argparse
import os
import datetime
import numpy as np
import regex as re
import __main__
import matplotlib.pyplot as plt
from scipy.special import comb
import itertools
from collections import defaultdict

import tabpfn_with_nats_bench.utils.get_important_lines as gil

# default names for zero-cost features
ZERO_COST_FEATURES_NAMES = {
    'f': 'Fisher',
    'g_n': 'GradNorm',
    'g': 'Grasp',
    'j_c': 'JacobCov',
    'p': 'Plain',
    's': 'SNIP',
    'sy': 'SynFlow',
    'sy_bn': 'SynFlowBN'
}

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


def get_single_seed_paths_to_logs(root_dir):
    """Get all log paths from a single seed directory."""
    log_paths = []
    for subdir in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, subdir)
        if os.path.isdir(folder_path):
            log_path = os.path.join(folder_path, "logs.txt")
            if os.path.isfile(log_path):
                log_paths.append(log_path)
    return log_paths

def get_multiple_seed_paths_to_logs(root_dir):
    """Get all log paths from multiple seed directories."""
    log_paths = []
    for seed_dir in os.listdir(root_dir):
        seed_path = os.path.join(root_dir, seed_dir)
        if os.path.isdir(seed_path):
            log_paths.extend(get_single_seed_paths_to_logs(seed_path))
    return log_paths

def extract_key_and_metrics_from_log(log_path: str, key_extractor_func: callable):
    """Extract metrics from a log file using a key extractor function."""
    if not os.path.isfile(log_path):
        raise ValueError(f"Expected log file: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # find the key using the provided function
    key = key_extractor_func(log_content)

    # find mean squared error
    mse_value = gil.find_mse(log_content)

    # find spearman's rank correlation coefficient
    spearman_value = gil.find_spearman(log_content)

    # find kendall's tau
    kendall_value = gil.find_kendall(log_content)

    return key, mse_value, spearman_value, kendall_value

def fill_metric_tables_from_logs(log_paths: list[str], key_extractor_func: callable):
    """Fill metric tables from a list of log files using a key extractor function."""
    mse_table = defaultdict(list)
    spearman_table = defaultdict(list)
    kendall_table = defaultdict(list)

    for log_path in log_paths:
        key, mse_value, spearman_value, kendall_value = extract_key_and_metrics_from_log(log_path, key_extractor_func)
        mse_table[key].append(mse_value)
        spearman_table[key].append(spearman_value)
        kendall_table[key].append(kendall_value)

    return mse_table, spearman_table, kendall_table

def format_zero_cost_combination_name(zcf_tuple):
    """Format the zero-cost combination name for printing."""
    if not zcf_tuple:
        return "No Zero-Cost Features"
    else:
        return ", ".join([ZERO_COST_FEATURES_NAMES[feat] for feat in zcf_tuple])

def sort_and_print_results(metric_table, metric_name, name_format_func=str, sort_ascending=True):
    sorted_results = []
    for key, metric_values in metric_table.items():
        name = name_format_func(key)
        metric_mean_value = np.mean(metric_values)
        metric_variance_value = np.var(metric_values) if len(metric_values) > 1 else 0.0
        sorted_results.append((name, metric_mean_value, metric_variance_value))
    
    sorted_results.sort(key=lambda x: x[1], reverse=not sort_ascending)

    print(f"\n=== {metric_name} Comparison ===")
    for name, value, var in sorted_results:
        if var > 0:
            print(f"{name}: {value:.6f} (Variance: {var:.6f})")
        else:
            print(f"{name}: {value:.6f}")

    best = sorted_results[0]
    print(f"\nBest {metric_name}: {best[0]} ({best[1]:.6f})")

def plot_binomial_groups(data, color='royalblue', title="Bars Grouped by Binomial Coefficients"):
    """
    Plots a dense bar chart grouped by n-choose-k combinations.
    Automatically detects 'n' based on the length of the data (len(data) must be a power of 2).
    """
    data = np.asarray(data)

    # Dynamically find 'n'. For 256 data points, n will be 8.
    n = int(np.log2(len(data)))

    # Safety check to ensure the data length is exactly 2^n
    if len(data) != 2**n:
        raise ValueError(f"Data length must be a power of 2. You provided {len(data)} items.")
    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(15, 4))
    x = np.arange(len(data))

    # Plot the bars
    ax.bar(x, data, width=1.0, edgecolor='none', color=color, alpha=0.8)

    # Calculate segment lengths using scipy instead of math
    k_values = np.arange(n + 1)
    segment_lengths = [comb(n, k, exact=True) for k in k_values]

    # Calculate exact boundaries and centers
    edges = np.insert(np.cumsum(segment_lengths), 0, 0) - 0.5
    tick_positions = []
    for i in range(len(segment_lengths)):
        center = edges[i] + (segment_lengths[i] / 2.0)
        tick_positions.append(center)

    # Apply ticks and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(k) for k in k_values])

    # Draw vertical dividing lines
    for edge in edges[1:-1]:
        ax.axvline(x=edge, color='black', linestyle=':', alpha=0.4)

    # Clean up aesthetics
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(title, pad=15)
    plt.tight_layout()

    return fig, ax

def plot_bar_plot(metric_table, metric_name):
    # we will plot the data from noZC, then all combinations of 1, 2, 3, 4, 5, 6, 7, 8 zero-cost features
    # all together it will be 1 + 8 + 28 + 56 + 70 + 56 + 28 + 8 + 1 = 256 bars
    # we will group the bars by the number of zero-cost features used
    # we will use the function plot_binomial_groups to plot the data

    # start with the no zero-cost features case
    no_zc_values = metric_table.get((), [])
    assert no_zc_values != [], "No zero-cost features case not found in metric_table"

    best_min = np.inf
    best_min_combination = None
    best_max = -np.inf
    best_max_combination = None

    result_values = [np.mean(no_zc_values)]
    for r in range(1, len(ZERO_COST_FEATURES_NAMES) + 1):
        for combination in itertools.combinations(ZERO_COST_FEATURES_NAMES.keys(), r):
            # get the mean value of the metric for this combination
            metric_values = metric_table.get(tuple(sorted(combination)), [])
            assert metric_values != [], f"Combination {combination} not found in metric_table"
            
            metric_mean_value = np.mean(metric_values)
            if metric_mean_value < best_min:
                best_min = metric_mean_value
                best_min_combination = combination
            if metric_mean_value > best_max:
                best_max = metric_mean_value
                best_max_combination = combination
            result_values.append(metric_mean_value)

    # plot the results
    fig, ax = plot_binomial_groups(result_values, title=f"")

    # create horizontal lines for the best values of each metric
    if metric_name == "Mean Squared Error":
        ax.axhline(y=best_min, color='black', linestyle='--', label=f'Nejlepší hodnota: {best_min:.3f} ({format_zero_cost_combination_name(best_min_combination)})')
        ax.set_ylabel("Střední čtvercová chyba (MSE)", fontsize=16)
    elif metric_name == "Spearman's Rank Correlation Coefficient":
        ax.axhline(y=best_max, color='gray', linestyle='--', label=f'Nejlepší hodnota: {best_max:.3f} ({format_zero_cost_combination_name(best_max_combination)})')
        ax.set_ylabel("Spearmanův korelační koeficient ρ ", fontsize=16)
        ax.set_ylim(0.9, 0.98)
    elif metric_name == "Kendall's Tau (rank correlation coefficient)":
        ax.axhline(y=best_max, color='gray', linestyle='--', label=f'Nejlepší hodnota: {best_max:.3f} ({format_zero_cost_combination_name(best_max_combination)})')
        ax.set_ylabel("Kendallův korelační koeficient τ", fontsize=16)
        ax.set_ylim(0.76, 0.89)

    ax.set_xlabel("Počet použitých zero-cost proxy metrik", fontsize=16)
    ax.legend(loc='lower right', fontsize=16)

    # set the x-ticks and y-ticks font size
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.savefig(f"tabpfn_with_nats_bench/plots/{metric_name.replace(' ', '_')}_bar_plot.pdf", dpi=300, bbox_inches='tight')


