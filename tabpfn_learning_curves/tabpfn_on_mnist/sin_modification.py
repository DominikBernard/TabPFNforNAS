import argparse
import numpy as np

from tabpfn_learning_curves.utils.mnist_pipeline import load_learning_curve_data, split_data_compute_predictions_and_metrics, plot_dual_learning_curves, SCALING_FACTOR

parser = argparse.ArgumentParser(description="Run a single experiment for predicting learning curves with sin modification method using TabPFNRegressor.")
parser.add_argument("--log_file_path", type=str, default="tabpfn_learning_curves/MNIST_example_runs/at=mlp,bs=256,cs=,d=cosine,e=100,hls=32,lr=0.001,lrf=0.0005,m=0.005,o=SGD,s=42,i=2/logs.txt", help="Path to the log file containing learning curve data.")
parser.add_argument("--splitting_epoch", type=int, default=20, help="Epoch at which to split the data into training and testing sets.")
parser.add_argument("--sine_period", type=float, default=6*np.pi, help="Period for the sine transformation applied to the learning curve data.")
parser.add_argument("--path_to_save", type=str, default=None, help="Path to save the plot. If not provided, the plot will be displayed.")
parser.add_argument("--mode", type=str, default="sin_multiplication", choices=["sin_addition", "sin_multiplication"], help="Mode of operation: 'sin_addition' or 'sin_multiplication'.")



def addition_and_subtraction_function():
    add = lambda x, y: x + y
    subtract = lambda x, y: x - y
    return add, subtract

def multiplication_and_division_function():
    multiply = lambda x, y: x * y / SCALING_FACTOR
    divide = lambda x, y: x * SCALING_FACTOR / y
    return multiply, divide


if __name__ == "__main__":
    args = parser.parse_args()
    splitting_epoch = args.splitting_epoch

    # Select the appropriate transformation functions based on the mode
    if args.mode == "sin_addition":
        print("Using sine addition transformation.")
        transform_function, inverse_transform_function = addition_and_subtraction_function()
    elif args.mode == "sin_multiplication":
        print("Using sine multiplication transformation.")
        transform_function, inverse_transform_function = multiplication_and_division_function()

    # Load learning curve data
    X, y = load_learning_curve_data(args.log_file_path)

    # create new sine feature
    feature = SCALING_FACTOR * np.sin(X[:, 0] * 2 * np.pi / args.sine_period)

    # add the sine feature to X (as a new column)
    X = np.hstack((X, feature[:, np.newaxis]))

    # add the sine feature to y (as addition or multiplication)
    y_transformed = transform_function(y, feature)

    # Compute predictions and metrics
    y_transformed_pred, mse, last_pred_squared_error = split_data_compute_predictions_and_metrics(X, y_transformed, splitting_epoch)

    print(f"Mean Squared Error (before transformation): {mse}")
    print(f"Squared Error of Last Prediction (before transformation): {last_pred_squared_error}")

    # transform both y_transformed and y_transformed_pred back to the original space
    y_orig = inverse_transform_function(y_transformed, feature)
    y_pred = inverse_transform_function(y_transformed_pred, feature[splitting_epoch:])

    # if feature is really close to 0, we get big rounding error. To avoid it, we delete the values for which it happens.
    # close_to_zero_mask = np.isclose(feature, 0, atol=5)
    # y_orig = y_orig[~close_to_zero_mask]
    # y_pred = y_pred[~close_to_zero_mask[splitting_epoch:]]

    # compute the mean squared error after transforming back
    mse_after_transform = np.mean((y_pred - y_orig[splitting_epoch:]) ** 2)
    print(f"Mean Squared Error after transforming back: {mse_after_transform}")
    print(f"Squared Error of Last Prediction after transforming back: {(y_pred[-1] - y_orig[-1]) ** 2}")

    if args.mode == "sin_addition":
         # check works for addition and subtraction, but not for multiplication and division (it is a scale)
        assert np.isclose(mse_after_transform, mse), "MSE should be the same after transforming back, since the transformation is just a shift"
    
    # plot the results for visualization
    plot_dual_learning_curves(X[:, 0], X, y_transformed, y_transformed_pred, X, y_orig, y_pred, splitting_epoch, feature_names=[f"Sinus s periodou 6π"], path_to_save=args.path_to_save)






    
    





