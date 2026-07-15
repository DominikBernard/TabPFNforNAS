import argparse
import os
import numpy as np
import tabpfn
import json

from tabpfn_final_accuracy.utils.dataloader import skip_unconnected_archs_using_data, download_data_if_not_exists
from tabpfn_final_accuracy.utils.arch_features import get_features_and_label
from tabpfn_final_accuracy.utils.results_utils import verbose_meta_info, plot_one_scatter_plot
from tabpfn_final_accuracy.utils.best_archs import find_best_n_architecture

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
parser.add_argument('--plot', action='store_true', default=False, help='Plot predictions vs true labels after each epoch')
parser.add_argument('--plot_dir', type=str, default='tabpfn_in_whole_nas_algorithms/plots/', help='Directory to save plots if --plot is enabled, if --plot is enabled and --plot_dir is None, plots will be shown.')

parser.add_argument('--number_of_added_architectures_per_epoch', type=int, default=10, help='Number of architectures to add to the training set after each epoch')
parser.add_argument('--termination_condition_n_best', type=int, default=10, help='Termination condition: stop when at least one of the top "number_of_added_architectures_per_epoch" predictions is among the top n true labels')

def main(args: argparse.Namespace):
    """
    Main function to run the active learning loop using TabPFN.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # set random seed
    np.random.seed(args.seed)

    if args.verbose:
        verbose_meta_info(args)

    download_data_if_not_exists(args.datafile)

    with open(args.datafile, 'r') as f:
        data = json.load(f)

    # total_best_archs format: list of tuples (arch_index, test_accuracy)
    total_best_archs = find_best_n_architecture(data, args.termination_condition_n_best)
    total_best_arch_indices, _ = zip(*total_best_archs)

    if args.skip_unconnected:
        all_indices = np.arange(0, 15625)
        indices = skip_unconnected_archs_using_data(data, all_indices)
        if args.verbose:
            print(f"Filtered unconnected architectures: {len(all_indices) - len(indices)} removed, {len(indices)} remaining.")
    else:
        indices = np.arange(0, 15625)

    features_and_labels_for_all_archs = [get_features_and_label(data, int(i), args.connected_feature, args.arch_index_feature, args.edge_features, args.zero_cost_features, args.epochs_n_features) for i in range(15625)]
    features, labels = zip(*features_and_labels_for_all_archs)

    features = np.array(features)
    labels = np.array(labels)

    train_indices = np.random.choice(indices, args.train_size, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    epoch = 1

    if len(set(total_best_arch_indices).intersection(set(train_indices))) > 0:
        print("The initial training set already contains at least one of the top n architectures.")
        print(f"The goal was achieved at the end of the 0. epoch.")
        return

    while True:
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        test_features = features[test_indices]
        test_labels = labels[test_indices]

        # load tabpfn regression model
        model = tabpfn.TabPFNRegressor()
        model.fit(train_features, train_labels)

        # make predictions
        preds = model.predict(test_features)

        # evaluate
        mse = np.mean((preds - test_labels) ** 2)
        if args.verbose:
            print(f"Epoch {epoch}: Mean Squared Error: {mse}")

        # plot predictions vs true labels
        if args.plot:
            if args.plot_dir is not None:
                # create the plot directory if it doesn't exist
                os.makedirs(args.plot_dir, exist_ok=True)
                plot_one_scatter_plot(test_labels, preds, output_file=f"{args.plot_dir}/predictions_vs_true_epoch_{epoch}.pdf")
            else:
                plot_one_scatter_plot(test_labels, preds)

        # take the 10 highest predictions
        top_10_pred_indices = test_indices[np.argsort(preds)[-args.number_of_added_architectures_per_epoch:]]

        # check if any of the top 10 predictions are among the top 10 true labels
        # top_10_true_indices = np.argsort(test_labels)[-10:]
        best_archs_intersection = set(top_10_pred_indices).intersection(set(total_best_arch_indices))
        if len(best_archs_intersection) > 0:
            # find the winning architecture
            winning_archs_accuracies = [data[int(i)]["testacc"] for i in best_archs_intersection]
            winning_accuracy = np.max(winning_archs_accuracies)
            print(f"Intersection size: {len(best_archs_intersection)}")
            print(f"Indices of intersection: {best_archs_intersection}")
            print(f"Accuracies of intersection: {winning_archs_accuracies}")
            print(f"Maximal accuracy in intersection: {winning_accuracy}")

            print("Stopping criterion met: at least one of the top 10 predictions is among the top 10 true labels.")
            print(f"The goal was achieved at the end of the {epoch}. epoch -> {(epoch + 1) * 10} architectures trained.")
            break
        else:
            train_indices = np.concatenate((train_indices, top_10_pred_indices))
            test_indices = np.setdiff1d(indices, train_indices)
            epoch += 1
            if epoch > 20:
                print("Stopping criterion not met after 20 epochs. Stopping the process.")
                break

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main(main_args)

