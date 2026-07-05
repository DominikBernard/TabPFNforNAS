import argparse
import numpy as np
import tabpfn
import json

from tabpfn_final_accuracy.utils.dataloader import skip_unconnected_archs_using_data
from tabpfn_final_accuracy.utils.arch_features import get_features_and_label
from tabpfn_final_accuracy.utils.results_utils import verbose_meta_info, plot_predictions_vs_true

parser = argparse.ArgumentParser(description='Get predictions from tabpfn')

parser.add_argument('--train_size', type=int, default=10, help='Number of training samples to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Dataset to use')
parser.add_argument('--connected_feature', action='store_true', default=False, help='Add a feature indicating if the architecture is connected')
parser.add_argument('--arch_index_feature', action='store_true', default=False, help='Add the architecture index as a feature')
parser.add_argument('--edge_features', choices=['onehot', 'raw', 'none'], default='raw', help='How to include edge features: onehot, raw, or none')
parser.add_argument('--zero_cost_features',
                    nargs="*", choices=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'],
                    default=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'],
                    help='Zero-cost features to include, choose from: f (fisher), g_n (grad_norm), g (grasp), j_c (jacob_cov), p (plain), s (snip), sy (synflow), sy_bn (synflow_bn)') 
parser.add_argument('--epochs_n_features', type=int, default=0, help='Include first n training accuracies as features')
parser.add_argument('--skip_unconnected', action='store_true', default=True, help='Skip unconnected architectures')
parser.add_argument('--datafile', type=str, default='tabpfn_final_accuracy/data/merged_data.json', help='Path to the JSON file containing architecture data')
parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')


def main(args: argparse.Namespace):
    # set random seed
    np.random.seed(args.seed)

    with open(args.datafile, 'r') as f:
        data = json.load(f)

    if args.skip_unconnected:
        all_indices = np.arange(0, 15625)
        indices = skip_unconnected_archs_using_data(data, all_indices)
        if args.verbose:
            print(f"Filtered unconnected architectures: {len(all_indices) - len(indices)} removed, {len(indices)} remaining.")
    else:
        indices = np.arange(0, 15625)

    features_and_labels = [get_features_and_label(data, int(i), args.connected_feature, args.arch_index_feature, args.edge_features, args.zero_cost_features, args.epochs_n_features) for i in range(15625)]
    features, labels = zip(*features_and_labels)

    features = np.array(features)
    labels = np.array(labels)

    train_indices = np.random.choice(indices, args.train_size, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    epoch = 1

    while True:
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        test_features = features[test_indices]
        test_labels = labels[test_indices]
        
        if args.verbose:
            verbose_meta_info(args)
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

        # plot predictions vs true labels
        plot_predictions_vs_true(test_labels, preds)

        # take the 10 highest predictions
        top_10_pred_indices = np.argsort(preds)[-10:]

        # check if any of the top 10 predictions are among the top 10 true labels
        top_10_true_indices = np.argsort(test_labels)[-10:]
        if len(set(top_10_pred_indices).intersection(set(top_10_true_indices))) > 0:
            print("Stopping criterion met: at least one of the top 10 predictions is among the top 10 true labels.")
            print(f"The goal was achieved at the end of the {epoch}. epoch.")
            break
        else:
            train_indices = np.concatenate((train_indices, top_10_pred_indices))
            test_indices = np.setdiff1d(indices, train_indices)
            epoch += 1

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main(main_args)

