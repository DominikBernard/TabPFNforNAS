import numpy as np
import os
import argparse
from tabpfn_final_accuracy.utils.results_utils import get_multiple_seed_paths_to_logs, fill_metric_tables_from_logs, sort_and_print_results
import tabpfn_final_accuracy.utils.get_important_lines as gil


parser = argparse.ArgumentParser(description="Analyze results from multiple runs of experiments with different numbers of epochs as features.")
parser.add_argument("--root_dir", type=str, default="tabpfn_final_accuracy/logs/epochs_n_features/skipUnconnected/withZCF/", help="Root directory containing experiment logs.")

def enf_key_extractor_func(log_content):
    """Extract the epochs_n_features key from the log content."""
    enf_value = gil.find_epochs_n_features(log_content)
    return enf_value

def name_format_func(key):
    """Format the epochs_n_features key for display."""
    # print one space before one digit numbers
    if key < 10:
        return f" {key} Epochs as Features"
    return f"{key} Epochs as Features"

if __name__ == "__main__":
    args = parser.parse_args()
    root_dir = args.root_dir

    log_paths = get_multiple_seed_paths_to_logs(root_dir)

    mse_table, spearman_table, kendall_table = fill_metric_tables_from_logs(log_paths, enf_key_extractor_func)
    
    # sort by MSE
    sort_and_print_results(mse_table, "Mean Squared Error", name_format_func=name_format_func, sort_ascending=True)

    # sort by Spearman's Rank Correlation Coefficient
    sort_and_print_results(spearman_table, "Spearman's Rank Correlation Coefficient", name_format_func=name_format_func, sort_ascending=False)

    # sort by Kendall's Tau
    sort_and_print_results(kendall_table, "Kendall's Tau (rank correlation coefficient)", name_format_func=name_format_func, sort_ascending=False)
