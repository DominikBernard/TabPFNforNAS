import os
from collections import defaultdict
import argparse

import tabpfn_final_accuracy.utils.get_important_lines as gil
from tabpfn_final_accuracy.utils.results_utils import plot_bar_plot, sort_and_print_results, format_zero_cost_combination_name, fill_metric_tables_from_logs, get_single_seed_paths_to_logs, get_multiple_seed_paths_to_logs

parser = argparse.ArgumentParser(description="Analyze results from multiple runs of zero-cost combination experiments.")
parser.add_argument("--root_dir", type=str, default="tabpfn_final_accuracy/logs/zero_cost_combination/skipUnconnected/seed_42", help="Root directory containing experiment logs.")
parser.add_argument("--plot", action="store_true", default=False, help="Whether to plot the results.")
parser.add_argument("--multiple_seeds", action="store_true", default=False,
                    help="Whether the root_dir contains multiple seeds (subdirectories). If not, it is assumed that the root_dir contains folders with each combination of zero-cost features directly.")


def zcf_key_extractor_func(log_content):
    """Extract the zero-cost combination key from the log content."""
    zcf_list = gil.find_zero_cost_features(log_content)
    return tuple(sorted(zcf_list))  # Sort to ensure consistent ordering

if __name__ == "__main__":
    args = parser.parse_args()
    root_dir = args.root_dir

    # Collect all log paths from the root directory
    if args.multiple_seeds:
        log_paths = get_multiple_seed_paths_to_logs(root_dir)
    else:
        log_paths = get_single_seed_paths_to_logs(root_dir)

    # Fill metric tables from logs
    mse_table, spearman_table, kendall_table = fill_metric_tables_from_logs(log_paths, zcf_key_extractor_func)

    metrics_to_analyze = [
        ("Mean Squared Error", mse_table, True),
        ("Spearman's Rank Correlation Coefficient", spearman_table, False),
        ("Kendall's Tau (rank correlation coefficient)", kendall_table, False)
    ]

    for metric_name, metric_table, sort_ascending in metrics_to_analyze:
        # sort by the current metric
        sort_and_print_results(metric_table, metric_name, name_format_func=format_zero_cost_combination_name, sort_ascending=sort_ascending)
        if args.plot:
            plot_bar_plot(metric_table, metric_name)

