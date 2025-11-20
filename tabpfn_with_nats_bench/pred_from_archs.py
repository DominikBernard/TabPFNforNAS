import nats_bench
import argparse
import numpy as np
import tabpfn
import matplotlib.pyplot as plt
import os

from change_logdir import change_logdir_if_default
from dataloader import create_dataset_from_api

parser = argparse.ArgumentParser(description='Get predictions from tabpfn')

parser.add_argument('--train_size', type=int, default=1_000, help='Number of training samples to use')
parser.add_argument('--test_size', type=int, default=200, help='Number of test samples to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Dataset to use')
parser.add_argument('--connected_feature', action='store_true', default=False, help='Add a feature indicating if the architecture is connected')
parser.add_argument('--arch_index_feature', action='store_true', default=False, help='Add the architecture index as a feature')
parser.add_argument('--edge_features_onehot', action='store_true', default=False, help='Use one-hot encoding for edge features')
parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
parser.add_argument('--logdir', type=str, default='logs', help='Directory to save logs and plots, if default, a timestamped directory will be created')

def main(args: argparse.Namespace, api: nats_bench.NATStopology):
    # set random seed
    np.random.seed(args.seed)

    change_logdir_if_default(args)

    chosen_indices, test_features, test_labels = create_dataset_from_api(np.arange(0, 15625), args, api, test_or_train="test")
    remaining_indices = np.setdiff1d(np.arange(0, 15625), chosen_indices)
    _, train_features, train_labels = create_dataset_from_api(remaining_indices, args, api, test_or_train="train")
    
    if args.verbose:
        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test labels shape: {test_labels.shape}")

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

