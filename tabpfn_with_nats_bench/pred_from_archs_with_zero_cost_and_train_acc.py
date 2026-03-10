import argparse
import numpy as np
import tabpfn
import os
import gdown
import json
import scipy.stats as stats

from utils.results_utils import change_logdir_if_default, plot_predictions_vs_true, verbose_meta_info
from utils.dataloader import create_dataset_from_data, skip_unconnected_archs_using_data

parser = argparse.ArgumentParser(description='Get predictions from tabpfn')

parser.add_argument('--train_size', type=int, default=1_000, help='Number of training samples to use')
parser.add_argument('--test_size', type=int, default=200, help='Number of test samples to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Dataset to use')
parser.add_argument('--connected_feature', action='store_true', default=False, help='Add a feature indicating if the architecture is connected')
parser.add_argument('--arch_index_feature', action='store_true', default=False, help='Add the architecture index as a feature')
parser.add_argument('--edge_features', choices=['onehot', 'raw', 'none'], default='raw', help='How to include edge features: onehot, raw, or none')
parser.add_argument('--zero_cost_features',
                    nargs="*", choices=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'],
                    default=[],
                    help='Zero-cost features to include, choose from: f (fisher), g_n (grad_norm), g (grasp), j_c (jacob_cov), p (plain), s (snip), sy (synflow), sy_bn (synflow_bn)') 
parser.add_argument('--epochs_n_features', type=int, default=0, help='Include first n training accuracies as features')
parser.add_argument('--skip_unconnected', action='store_true', default=False, help='Skip unconnected architectures')
parser.add_argument('--datafile', type=str, default='tabpfn_with_nats_bench/data/merged_data.json', help='Path to the JSON file containing architecture data')
parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')
parser.add_argument('--logdir', type=str, default='tabpfn_with_nats_bench/logs', help='Directory to save logs and plots, if default, a timestamped directory will be created')


def main(args: argparse.Namespace):
    # set random seed
    np.random.seed(args.seed)

    change_logdir_if_default(args)

    # check if datafile exists
    if not os.path.exists(args.datafile):
        print(f"Data file {args.datafile} does not exist, downloading it from google drive...")

        # create the directory if it does not exist
        if not os.path.exists(os.path.dirname(args.datafile)):
            os.makedirs(os.path.dirname(args.datafile), exist_ok=True)

        # download them from google drive
        # Your File ID from the link
        url = 'https://drive.google.com/uc?id=1htCa-ztKlkkYyje6K98q8n-CioPUA_Pk'
        gdown.download(url, args.datafile, quiet=False)

    with open(args.datafile, 'r') as f:
        data = json.load(f)

    if args.skip_unconnected:
        all_indices = np.arange(0, 15625)
        indices = skip_unconnected_archs_using_data(data, all_indices)
        if args.verbose:
            print(f"Filtered unconnected architectures: {len(all_indices) - len(indices)} removed, {len(indices)} remaining.")
    else:
        indices = np.arange(0, 15625)

    chosen_test_indices, test_features, test_labels = create_dataset_from_data(indices, args, data, test_or_train="test")
    remaining_indices = np.setdiff1d(indices, chosen_test_indices)
    _, train_features, train_labels = create_dataset_from_data(remaining_indices, args, data, test_or_train="train")

    if args.verbose:
        verbose_meta_info(args)
        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test labels shape: {test_labels.shape}")

    # assert no value in features and labels is nan
    assert not np.isnan(train_features).any(), "Train features contain NaN values, check your data preprocessing, it is necessary to replace them with some value, e.g. 0, before training the model"
    assert not np.isnan(test_features).any(), "Test features contain NaN values, check your data preprocessing, it is necessary to replace them with some value, e.g. 0, before training the model"
    assert not np.isnan(train_labels).any(), "Train labels contain NaN values, check your data preprocessing, it is necessary to replace them with some value, e.g. 0, before training the model"
    assert not np.isnan(test_labels).any(), "Test labels contain NaN values, check your data preprocessing, it is necessary to replace them with some value, e.g. 0, before training the model"


    # load tabpfn regression model
    model = tabpfn.TabPFNRegressor()
    model.fit(train_features, train_labels)

    # make predictions
    preds = model.predict(test_features)
    if args.verbose:
        print(f"Predictions shape: {preds.shape}")

    # evaluate
    mse = np.mean((preds - test_labels) ** 2)
    print(f"Mean Squared Error: {mse}")

    # compute Spearman's rank correlation coefficient
    spearman_corr, _ = stats.spearmanr(preds, test_labels)
    print(f"Spearman's Rank Correlation Coefficient: {spearman_corr}")

    # compute kendall's tau (rank correlation coefficient)
    kendall_corr, _ = stats.kendalltau(preds, test_labels)
    print(f"Kendall's Tau (rank correlation coefficient): {kendall_corr}")

    # store predictions in the text file
    with open(os.path.join(args.logdir, "predictions.txt"), "w") as f:
        f.write("Arch index, Prediction\n")
        for idx, pred in zip(chosen_test_indices, preds):
            f.write(f"{idx}, {pred}\n")

    # plot the results
    plot_predictions_vs_true(test_labels, preds, args.logdir)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main(main_args)

