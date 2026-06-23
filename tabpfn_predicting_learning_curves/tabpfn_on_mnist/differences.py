import argparse
import numpy as np

from tabpfn_predicting_learning_curves.utils.mnist_pipeline import load_learning_curve_data, split_data_compute_predictions_and_metrics, plot_dual_learning_curves

parser = argparse.ArgumentParser(description="Run a single experiment for predicting learning curves with difference method using TabPFNRegressor.")
parser.add_argument("--log_file_path", type=str, default="tabpfn_predicting_learning_curves/MNIST_example_runs/run2.txt", help="Path to the log file containing learning curve data.")
parser.add_argument("--splitting_epoch", type=int, default=20, help="Epoch at which to split the data into training and testing sets.")
parser.add_argument("--path_to_save", type=str, default="tabpfn_predicting_learning_curves/plots/curve_modification/run2/differences.pdf", help="Path to save the plot.")


if __name__ == "__main__":
    print("Running the experiment with differences as target values.")

    args = parser.parse_args()
    splitting_epoch = args.splitting_epoch

    # Load learning curve data
    X, y = load_learning_curve_data(args.log_file_path)

    # Compute differences in accuracy as target values
    y_diff = np.diff(np.hstack([0, y]))

    # Compute predictions and metrics
    y_diff_pred, mse, last_pred_squared_error = split_data_compute_predictions_and_metrics(X, y_diff, splitting_epoch)

    print(f"Mean Squared Error (before transformation): {mse}")
    print(f"Squared Error of Last Prediction (before transformation): {last_pred_squared_error}")

    # convert differences back to actual values
    y_orig = np.cumsum(y_diff)  # Cumulative sum to get actual accuracy values
    y_pred = np.cumsum(y_diff_pred) + y_orig[splitting_epoch] # Cumulative sum of predictions

    mse_after_cumsum = np.mean((y_pred - y_orig[splitting_epoch:]) ** 2)
    print(f"Mean Squared Error after cumulative sum: {mse_after_cumsum}")
    print(f"Squared Error of Last Prediction after cumulative sum: {(y_pred[-1] - y_orig[-1]) ** 2}")

    # plot the results for visualization
    plot_dual_learning_curves(X[:, 0], X, y_diff, y_diff_pred, X, y_orig, y_pred, splitting_epoch, feature_names=None, path_to_save=args.path_to_save)
    

