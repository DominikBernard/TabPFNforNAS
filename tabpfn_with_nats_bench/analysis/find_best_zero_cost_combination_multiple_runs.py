import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools

import tabpfn_with_nats_bench.utils.get_important_lines as gil
from tabpfn_with_nats_bench.utils.results_utils import plot_binomial_groups

def sort_and_print_results(metric_table, metric_name, sort_ascending=True):
    sorted_results = []
    for zcf_tuple, metric_values in metric_table.items():
        name = ", ".join([zero_cost_features_names[feat] for feat in zcf_tuple]) if zcf_tuple else "No Zero-Cost Features"
        metric_mean_value = np.mean(metric_values)
        metric_variance_value = np.var(metric_values)
        sorted_results.append((name, metric_mean_value, metric_variance_value))
    sorted_results.sort(key=lambda x: x[1], reverse=not sort_ascending)

    print(f"\n=== {metric_name} Comparison ===")
    for name, value, var in sorted_results:
        print(f"{name}: {value:.6f} (Variance: {var:.6f})")

    best = sorted_results[0]
    print(f"\nBest {metric_name}: {best[0]} ({best[1]:.6f})")

def plot_bar_plot(metric_table, metric_name, zero_cost_features, zero_cost_features_names):
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
    for r in range(1, len(zero_cost_features) + 1):
        for combination in itertools.combinations(zero_cost_features, r):
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
        ax.axhline(y=best_min, color='black', linestyle='--', label=f'Nejlepší hodnota: {best_min:.3f} ({", ".join([zero_cost_features_names[feat] for feat in best_min_combination])})')
        ax.set_ylabel("Střední čtvercová chyba (MSE)", fontsize=16)
    elif metric_name == "Spearman's Rank Correlation Coefficient":
        ax.axhline(y=best_max, color='gray', linestyle='--', label=f'Nejlepší hodnota: {best_max:.3f} ({", ".join([zero_cost_features_names[feat] for feat in best_max_combination])})')
        ax.set_ylabel("Spearmanův korelační koeficient ρ ", fontsize=16)
        ax.set_ylim(0.9, 0.98)
    elif metric_name == "Kendall's Tau (rank correlation coefficient)":
        ax.axhline(y=best_max, color='gray', linestyle='--', label=f'Nejlepší hodnota: {best_max:.3f} ({", ".join([zero_cost_features_names[feat] for feat in best_max_combination])})')
        ax.set_ylabel("Kendallův korelační koeficient τ", fontsize=16)
        ax.set_ylim(0.76, 0.89)

    ax.set_xlabel("Počet použitých zero-cost proxy metrik", fontsize=16)
    ax.legend(loc='lower right', fontsize=16)

    # set the x-ticks and y-ticks font size
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.savefig(f"tabpfn_with_nats_bench/plots/{metric_name.replace(' ', '_')}_bar_plot.pdf", dpi=300, bbox_inches='tight')

root_dir = "tabpfn_with_nats_bench/logs/zero_cost_combination/"

zero_cost_features = ['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn']
zero_cost_features_names = {
    'f': 'Fisher',
    'g_n': 'GradNorm',
    'g': 'Grasp',
    'j_c': 'JacobCov',
    'p': 'Plain',
    's': 'SNIP',
    'sy': 'SynFlow',
    'sy_bn': 'SynFlowBN'
}

paths_to_experiments = os.listdir(root_dir)
paths_to_experiments = [p for p in paths_to_experiments if p.endswith("_su")] # remove dirs that does not end with _su

mse_table = defaultdict(list)
spearman_table = defaultdict(list)
kendall_table = defaultdict(list)

for run_idx, exp_dir in enumerate(paths_to_experiments):
    path_to_exp_dir = os.path.join(root_dir, exp_dir)
    for subdir in os.listdir(path_to_exp_dir):
        folder_path = os.path.join(path_to_exp_dir, subdir)
        log_path = os.path.join(folder_path, "logs.txt")

        if not os.path.isdir(folder_path) or not os.path.isfile(log_path):
            raise ValueError(f"Expected directory with logs.txt: {folder_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        # find used zero-cost features
        zcf_list = gil.find_zero_cost_features(log_content)
        
        # find mean squared error
        mse_value = gil.find_mse(log_content)
        mse_table[tuple(sorted(zcf_list))].append(mse_value)

        # find spearman's rank correlation coefficient
        spearman_value = gil.find_spearman(log_content)
        spearman_table[tuple(sorted(zcf_list))].append(spearman_value)

        # find kendall's tau
        kendall_value = gil.find_kendall(log_content)
        kendall_table[tuple(sorted(zcf_list))].append(kendall_value)

# sort by MSE
sort_and_print_results(mse_table, "Mean Squared Error")
plot_bar_plot(mse_table, "Mean Squared Error", zero_cost_features, zero_cost_features_names)

# sort by Spearman's Rank Correlation Coefficient
sort_and_print_results(spearman_table, "Spearman's Rank Correlation Coefficient", sort_ascending=False)
plot_bar_plot(spearman_table, "Spearman's Rank Correlation Coefficient", zero_cost_features, zero_cost_features_names)

# sort by Kendall's Tau
sort_and_print_results(kendall_table, "Kendall's Tau (rank correlation coefficient)", sort_ascending=False)
plot_bar_plot(kendall_table, "Kendall's Tau (rank correlation coefficient)", zero_cost_features, zero_cost_features_names)
