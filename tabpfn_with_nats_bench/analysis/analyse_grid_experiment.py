import argparse
import os
import numpy as np
import tabpfn_with_nats_bench.utils.get_important_lines as gil
import regex as re
from collections import defaultdict

parser = argparse.ArgumentParser(description='Analyse grid search experiment results')
parser.add_argument('--logdir', type=str, default='tabpfn_with_nats_bench/logs/grid_search_results', help='Root directory where the grid search results are stored')
parser.add_argument('--train_size', type=int, default=1000, help='Size of the training set')
parser.add_argument('--analyse_connected_feature', action='store_true', default=True, help='Analyse the effect of the connected feature')
parser.add_argument('--analyse_arch_index_feature', action='store_true', default=True, help='Analyse the effect of the architecture index feature')
parser.add_argument('--analyse_edge_features', action='store_true', default=True, help='Analyse the effect of edge features')
parser.add_argument('--analyse_zero_cost_features', action='store_true', default=False, help='Analyse the effect of zero-cost features')
parser.add_argument('--analyse_skip_unconnected', action='store_true', default=True, help='Analyse the effect of skipping unconnected architectures')
parser.add_argument('--analyse_epochs_n_features', action='store_true', default=False, help='Analyse the effect of including first n training accuracies as features')
parser.add_argument('--verbose_warning', action='store_true', default=False, help='Print warnings for experiments where TabPFN was unable to make predictions')


"""
the structure of the logdir is expected to be as follows:
logdir/
    seed_1/
        all_experimental_dirs...
    seed_2/
        all_experimental_dirs...
    ...
"""

def find_all_log_file_paths(logdir):
    # check if logdir exists
    if not os.path.exists(logdir):
        raise ValueError(f"Log directory does not exist: {logdir}")
    
    log_file_paths = []
    
    # iterate over all seeds
    for seed_dir in os.listdir(logdir):
        seed_path = os.path.join(logdir, seed_dir)
        if not os.path.isdir(seed_path):
            continue
        
        # iterate over all experimental directories for this seed
        for exp_dir in os.listdir(seed_path):
            exp_path = os.path.join(seed_path, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            
            log_file_path = os.path.join(exp_path, "logs.txt")
            if os.path.isfile(log_file_path):
                log_file_paths.append(log_file_path)

    return log_file_paths

def filter_log_file_paths_by_args(log_file_paths, args: argparse.Namespace):
    # now we have all log file paths, we will remove files not matching the specified arguments
    # at first, we will filter by train_size
    filtered_log_file_paths = [path for path in log_file_paths if f"train-{args.train_size}_" in path]

    # now we will filter by the other arguments
    if not args.analyse_connected_feature:
        filtered_log_file_paths = [path for path in filtered_log_file_paths if "cf-False" in path]
    
    if not args.analyse_arch_index_feature:
        filtered_log_file_paths = [path for path in filtered_log_file_paths if "aif-False" in path]
    
    if not args.analyse_edge_features:
        filtered_log_file_paths = [path for path in filtered_log_file_paths if "ef-none" in path]
    
    if not args.analyse_zero_cost_features:
        filtered_log_file_paths = [path for path in filtered_log_file_paths if "noZC" in path]
    
    if not args.analyse_skip_unconnected:
        filtered_log_file_paths = [path for path in filtered_log_file_paths if "noSkipUnconnected" in path]
    
    if not args.analyse_epochs_n_features:
        filtered_log_file_paths = [path for path in filtered_log_file_paths if "enf-0" in path]

    return filtered_log_file_paths

def create_results_dictionary(filtered_log_file_paths, verbose_warning=False):
    # create dictionary to store results
    # the dictionary will have the following structure:
    # results = { key: { "mse": [], "spearman": [], "kendall": [] } }
    # key will be a string representing the combination of arguments, e.g. "cf-True_aif-False_ef-onehot_noZC_noSkipUnconnected_enf-0"
    results = defaultdict(lambda: {"mse": [], "spearman": [], "kendall": []})

    # create pattern to extract the key from the log file path
    pattern = re.compile(r"cf-(True|False)_aif-(True|False)_ef-(raw|onehot|none)_(noZC|withZC)_(noSkipUnconnected|skipUnconnected)_enf-(0|10)")

    # iterate over all filtered log file paths
    for log_file_path in filtered_log_file_paths:
        # create a key for this log file
        match = pattern.search(log_file_path)
        if not match:
            raise ValueError(f"Log file path does not match expected pattern: {log_file_path}")
        key = match.group(0)

        # read the log file
        with open(log_file_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        # find mean squared error
        mse_value = gil.find_mse(log_content)

        if mse_value is None:
            # TabPFN was unable to make predictions for this experiment, skip it
            if verbose_warning:
                print(f"Warning: TabPFN was unable to make predictions for this experiment, skipping it. Log file path: {log_file_path}")
            continue

        # find spearman's rank correlation coefficient
        spearman_value = gil.find_spearman(log_content)

        # find kendall's tau
        kendall_value = gil.find_kendall(log_content)

        # store the results
        results[key]["mse"].append(mse_value)
        results[key]["spearman"].append(spearman_value)
        results[key]["kendall"].append(kendall_value)

    return results

def print_results(results):

    # print the results
    for key, metrics in results.items():
        mse_mean = np.mean(metrics["mse"])
        mse_var = np.var(metrics["mse"])
        spearman_mean = np.mean(metrics["spearman"])
        spearman_var = np.var(metrics["spearman"])
        kendall_mean = np.mean(metrics["kendall"])
        kendall_var = np.var(metrics["kendall"])

        print(f"=== {key} ===")
        print(f"Mean Squared Error: {mse_mean:.6f} (Variance: {mse_var:.6f})")
        print(f"Spearman's Rank Correlation Coefficient: {spearman_mean:.6f} (Variance: {spearman_var:.6f})")
        print(f"Kendall's Tau (rank correlation coefficient): {kendall_mean:.6f} (Variance: {kendall_var:.6f})")

def create_results_table(results):
    # create a table with the results in format
    # the table will have the following structure:
    # key & mse_mean & spearman_mean & kendall_mean \\
    # filter only cf-False aif-False and noSkipUnconnected
    # or filter other combinations
    table = []
    for key, metrics in results.items():
        # if "cf-False" not in key or "aif-False" not in key or "noSkipUnconnected" not in key:
        #     continue
        # if "aif-True" not in key or "ef-raw" not in key:
        #     continue
        mse_mean = np.mean(metrics["mse"])
        spearman_mean = np.mean(metrics["spearman"])
        kendall_mean = np.mean(metrics["kendall"])

        table.append(f"{key} & {mse_mean:.6f} & {spearman_mean:.6f} & {kendall_mean:.6f} \\\\")

    return table

def main(args: argparse.Namespace):

    log_file_paths = find_all_log_file_paths(args.logdir)
    
    filtered_log_file_paths = filter_log_file_paths_by_args(log_file_paths, args)

    results = create_results_dictionary(filtered_log_file_paths, verbose_warning=args.verbose_warning)

    print_results(results)

    # create a table with the results
    print("\n=== Results Table ===")
    table = create_results_table(results)
    for row in table:
        print(row)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
