from tabpfn import TabPFNRegressor
import argparse
import numpy as np
import os

import tabpfn_predicting_learning_curves.utils.load_prepare_plot as lpp

parser = argparse.ArgumentParser(description="Predict learning curves using TabPFN on NATS-Bench data.")
parser.add_argument("--train_size", type=int, default=10, help="Number of training examples.")
parser.add_argument("--test_size", type=int, default=200, help="Number of test examples.")
parser.add_argument("--test_splitting_epoch", type=int, default=20, help="Epoch at which to split the target data into train and test sets.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--datapath", type=str, default="tabpfn_with_nats_bench/data/merged_data.json", help="Path to the JSON file containing the data.")
parser.add_argument("--path_to_dir_to_save_results", type=str, default="tabpfn_predicting_learning_curves/logs/results/", help="Path to save the results. If not provided, the results will be displayed.")
parser.add_argument("--sort_indices", default=True, action='store_true', help="Whether to sort the indices based on the test accuracy at the splitting epoch.")

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, test_archs_test: dict[int, tuple[list[list[int]], list[int]]], test_archs_train: dict[int, tuple[list[list[int]], list[int]]], implicit_test_labels_shape: tuple[int, int]) -> tuple[np.ndarray, float, np.ndarray, float]:
    """
    Train the TabPFNRegressor and evaluate its performance.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        test_archs_test (dict[int, tuple[list[list[int]], list[int]]]): Test architectures and their features/labels for test evaluation.
        test_archs_train (dict[int, tuple[list[list[int]], list[int]]]): Test architectures and their features/labels for training.
        implicit_test_labels_shape (tuple[int, int]): The implicit shape of the test labels for reshaping predictions.

    Returns:
        tuple: A tuple containing the predictions, the mean squared error on the test set, the mean squared error per architecture, and the mean squared error for the last epoch.
    """
    X_train_added = []
    y_train_added = []

    X_test = []
    y_test = []
    for arch_idx in test_archs_train:
        X_train_added.extend(test_archs_train[arch_idx][0])
        y_train_added.extend(test_archs_train[arch_idx][1])

        X_test.extend(test_archs_test[arch_idx][0])
        y_test.extend(test_archs_test[arch_idx][1])

    X_train = np.vstack((X_train, np.array(X_train_added)))
    y_train = np.hstack((y_train, np.array(y_train_added)))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = TabPFNRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    predictions_reshaped, whole_mse, mse_per_arch, mse_for_last_epoch = lpp.calc_results(predictions, y_test, implicit_test_labels_shape)

    return predictions_reshaped, whole_mse, mse_per_arch, mse_for_last_epoch

def main(args: argparse.Namespace):

    # fix random seed
    np.random.seed(args.seed)

    data = lpp.load_data(args.datapath)

    # Prepare dataset
    X_train, y_train, test_archs_test, test_archs_train, train_indices, test_indices, test_labels_implicit_shape = lpp.prepare_dataset(data, train_examples=args.train_size, test_examples=args.test_size, test_splitting_epoch=args.test_splitting_epoch, sort_indices=args.sort_indices)

    # Train and evaluate the model
    predictions, mse, mse_per_arch, mse_for_last_epoch = train_and_evaluate(X_train, y_train, test_archs_test, test_archs_train, test_labels_implicit_shape)

    # create directory to save results if it does not exist
    if args.path_to_dir_to_save_results and not os.path.exists(args.path_to_dir_to_save_results):
        os.makedirs(args.path_to_dir_to_save_results)

    lpp.plot_results(data, train_indices, test_indices, predictions, args.test_splitting_epoch, path_to_save=args.path_to_dir_to_save_results)
    lpp.store_results(test_indices, predictions, mse, mse_per_arch, mse_for_last_epoch, path_to_save=args.path_to_dir_to_save_results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

