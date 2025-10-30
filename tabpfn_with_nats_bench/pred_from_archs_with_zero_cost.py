import nats_bench
from arch_features import get_features_and_label
import argparse
import numpy as np
import tabpfn
import matplotlib.pyplot as plt
import os
import datetime
import regex as re
from typing import Literal
import json

parser = argparse.ArgumentParser(description='Get predictions from tabpfn')

parser.add_argument('--train_size', type=int, default=1_000, help='Number of training samples to use')
parser.add_argument('--test_size', type=int, default=200, help='Number of test samples to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Dataset to use')
parser.add_argument('--connected_feature', action='store_true', default=True, help='Add a feature indicating if the architecture is connected')
parser.add_argument('--arch_index_feature', action='store_true', default=False, help='Add the architecture index as a feature')
parser.add_argument('--edge_features_onehot', action='store_true', default=False, help='Use one-hot encoding for edge features')
parser.add_argument('--zero_cost_features',
                    nargs="*", choices=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'],
                    default=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'],
                    help='Zero-cost features to include, choose from: f (fisher), g_n (grad_norm), g (grasp), j_c (jacob_cov), p (plain), s (snip), sy (synflow), sy_bn (synflow_bn)') 
parser.add_argument('--datafile', type=str, default='data.json', help='Path to the JSON file containing architecture data')
parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')
parser.add_argument('--logdir', type=str, default='logs', help='Directory to save logs and plots, if default, a timestamped directory will be created')


def change_logdir_if_default(main_args: argparse.Namespace):
    """Change logdir if it is the default value."""
    # check if logdir is default
    if main_args.logdir == 'logs':
        # Create logdir name.
        main_args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(main_args).items())))
        ))
    os.makedirs(main_args.logdir, exist_ok=True)

def create_dataset(indices_to_choose_from: np.ndarray, args: argparse.Namespace, data: list[dict], test_or_train: Literal["test", "train"] = "train") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset of features and labels.
    Args:
        indices_to_choose_from: Indices to choose architectures from.
        args: Command line arguments.
        test_or_train: Whether to create test or train dataset.

    Returns:
        chosen_indices: Indices of chosen architectures.
        features: Features of chosen architectures.
        labels: Labels of chosen architectures.
    """
    size = args.test_size if test_or_train == "test" else args.train_size
    chosen_indices = np.random.choice(indices_to_choose_from, size, replace=False)

    features_and_labels = [get_features_and_label(data, i, args.connected_feature, args.arch_index_feature, args.edge_features_onehot, args.zero_cost_features) for i in chosen_indices]
    features, labels = zip(*features_and_labels)

    features = np.array(features)
    labels = np.array(labels)

    return chosen_indices, features, labels

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


def main(args: argparse.Namespace, api: nats_bench.NATStopology):
    # set random seed
    np.random.seed(args.seed)

    change_logdir_if_default(args)

    with open(args.datafile, 'r') as f:
        data = json.load(f)

    chosen_indices, test_features, test_labels = create_dataset(np.arange(0, 15625), args, data, test_or_train="test")
    remaining_indices = np.setdiff1d(np.arange(0, 15625), chosen_indices)
    _, train_features, train_labels = create_dataset(remaining_indices, args, data, test_or_train="train")

    if args.verbose:
        verbose_meta_info(args)
        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test labels shape: {test_labels.shape}")

    # assert no value in features and labels is nan
    assert not np.isnan(train_features).any(), "Train features contain NaN values"
    assert not np.isnan(test_features).any(), "Test features contain NaN values"
    assert not np.isnan(train_labels).any(), "Train labels contain NaN values"
    assert not np.isnan(test_labels).any(), "Test labels contain NaN values"


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
    
    # save the figure
    plt.savefig(os.path.join(args.logdir, "true_vs_pred.png"))
    # plt.show()

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    api = nats_bench.create(None, 'tss', fast_mode=True, verbose=False)

    assert isinstance(api, nats_bench.NATStopology), "API is not of type NATStopology" # assert api is tss type

    main(main_args, api)

