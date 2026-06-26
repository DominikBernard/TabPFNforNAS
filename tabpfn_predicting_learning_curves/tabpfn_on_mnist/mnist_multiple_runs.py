import argparse
from contextlib import redirect_stdout
import numpy as np

from tabpfn_predicting_learning_curves.utils.mnist_pipeline import prepare_multi_run_data, train_and_predict, compute_metrics, plot_multi_run_learning_curves, plot_multi_run_learning_curves_side_by_side
from tabpfn_predicting_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy
from tabpfn_predicting_learning_curves.utils.log_runs_to_tensors import extract_logs_from_subdirectories, create_tensors_from_logs

parser = argparse.ArgumentParser(description="Run an experiment on multiple learning curves with TabPFNRegressor.")
parser.add_argument("--log_dir", type=str, default="tabpfn_predicting_learning_curves/MNIST_example_runs", help="Directory containing the log files. If not provided, --log_paths will be used (default is 'tabpfn_predicting_learning_curves/MNIST_example_runs').")
parser.add_argument("--log_paths", type=str, nargs='+',
                    default=["tabpfn_predicting_learning_curves/MNIST_example_runs/mnist_training_diploma.py-2026-04-17_130608-at=mlp,bs=256,cs=,d=cosine,e=100,hls=32,lr=0.001,lrf=0.0005,m=0.005,o=SGD,r=False,s=42,t=1,i=2/logs.txt",
                             "tabpfn_predicting_learning_curves/MNIST_example_runs/mnist_training_diploma.py-2026-04-17_130608-at=cnn,bs=1024,cs=conv3x3-BN-relu-conv3x3-BN-relu-linear,d=linear,e=100,hls=32,lr=0.0005,lrf=1e-05,m=0.005,o=SGD,r=False,s=42,t=1,i=9/logs.txt"],
                    help="Paths to the log files containing learning curve data for multiple runs. ")
parser.add_argument("--splitting_epoch", type=int, default=20, help="Epoch at which to split the data into training and testing sets.")
parser.add_argument("--path_to_save_plot", type=str, default="tabpfn_predicting_learning_curves/plots/mnist_multiple_runs/all_runs.pdf", help="Path to save the plot. If not provided, the plot will be displayed.")
parser.add_argument("--test_run_index", type=int, default=-1, help="Index of the run to be used as the test run. Default is -1, which means EVERY run will be tested.")
parser.add_argument("--sort_by_accuracy", default=False, action='store_true', help="Whether to sort the runs by their accuracy (at the splitting epoch).")
parser.add_argument("--output_text_file", type=str, default="tabpfn_predicting_learning_curves/plots/mnist_multiple_runs/multi_runs.txt", help="Path to save the output text file. If not provided, the output will be printed to the console.")

def load_learning_curve_data(args: argparse.Namespace):
    """
    Load learning curve data from log files specified in the arguments.

    If --log_dir is provided, it will extract logs from the specified directory and create tensors from them.
    If --log_dir is not provided, it will directly extract epoch and dev accuracy from the log files specified in --log_paths.

    Args:
        args (argparse.Namespace): The command-line arguments containing log file paths or directory.
    
    Returns:
        learning_curve_data_list (list): A list of numpy arrays containing learning curve data for each run.
        indices (list or None): A list of indices that would sort the learning curve data by their accuracy at the splitting epoch, or None if not sorting.
    """
    indices = None
    if args.log_dir:
        learning_curve_data_list = create_tensors_from_logs(
        extract_logs_from_subdirectories(args.log_dir), acc_in_percent=True
    )
        if args.sort_by_accuracy:
            indices = np.argsort(np.argsort([learning_curve_data[args.splitting_epoch - 1, 1] for learning_curve_data in learning_curve_data_list]))
    else:
        learning_curve_data_list = [extract_epoch_dev_accuracy(path, acc_in_percent=True) for path in args.log_paths]
        if args.sort_by_accuracy:
            indices = [3, 10]

    return learning_curve_data_list, indices

def execute_experiment_for_one_test_run_index(test_run_index, learning_curve_data_list, splitting_epoch, custom_indices=None):
    print(f"Testing on run {test_run_index + 1}")
    X_train, y_train, X_test, y_test = prepare_multi_run_data(learning_curve_data_list, splitting_epoch, test_run_index=test_run_index, custom_indices=custom_indices)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Train the model and make predictions
    y_pred = train_and_predict(X_train, y_train, X_test)

    # Compute metrics
    mse, se_last = compute_metrics(y_pred, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"Squared Error at Last Epoch: {se_last}")

    return y_pred, mse, se_last

def main(args: argparse.Namespace):

    splitting_epoch = args.splitting_epoch
    test_run_index = args.test_run_index

    # Load learning curve data
    learning_curve_data_list, indices = load_learning_curve_data(args)
    if indices is None:
        indices = np.arange(len(learning_curve_data_list))  # Default indices if not sorting

    # Prepare dataset
    if test_run_index == -1:
        y_pred_list = []
        for test_run_index in range(len(learning_curve_data_list)):
            y_pred, mse, se_last = execute_experiment_for_one_test_run_index(test_run_index, learning_curve_data_list, splitting_epoch, custom_indices=indices)
            y_pred_list.append(y_pred)
        plot_multi_run_learning_curves_side_by_side(learning_curve_data_list, y_pred_list, splitting_epoch, path_to_save=args.path_to_save_plot)
    else:
        y_pred, mse, se_last = execute_experiment_for_one_test_run_index(test_run_index, learning_curve_data_list, splitting_epoch, custom_indices=indices)
        plot_multi_run_learning_curves(learning_curve_data_list, y_pred, splitting_epoch, test_run_index=test_run_index, path_to_save=args.path_to_save_plot)

    
if __name__ == "__main__":
    args = parser.parse_args()

    if args.output_text_file:
        with open(args.output_text_file, "w", encoding="utf-8") as f, redirect_stdout(f):
            main(args)
    else:
        main(args)

