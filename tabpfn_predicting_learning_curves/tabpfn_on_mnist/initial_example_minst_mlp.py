import numpy as np
import matplotlib.pyplot as plt
import argparse
from tabpfn import TabPFNRegressor

from tabpfn_predicting_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy
from tabpfn_predicting_learning_curves.utils.top_k_seasonalities import extract_top_k_seasonalities, build_fourier_features
from tabpfn_predicting_learning_curves.utils.find_function_opt_params import inverse_model, log_model, fit_and_evaluate

parser = argparse.ArgumentParser(description="Predict learning curves using TabPFN")
parser.add_argument('--splitting_epoch', type=int, default=20, help='Epoch at which to split train and test data')
parser.add_argument('--log_file_path', type=str, default="tabpfn_predicting_learning_curves/MNIST_example_runs/run1.txt", help='Path to the log file containing learning curve data')

# feature engineering functions

def add_trig_features(X, epochs):
    f1 = 100 * np.sin(epochs)[:, np.newaxis]
    f2 = 100 * np.cos(epochs)[:, np.newaxis]
    f3 = 100 * np.sin(epochs * np.pi / 200)[:, np.newaxis]
    f4 = 100 * np.cos(epochs * np.pi / 200)[:, np.newaxis]
    return np.hstack((X, f1, f2, f3, f4))

def add_inverse_feature(X, epochs):
    f = 100 * (0.9 - 1 / (epochs + 1))[:, np.newaxis]
    return np.hstack((X, f))

def add_optimal_inverse_feature(X, epochs, y):
    params, _ = fit_and_evaluate(inverse_model, epochs, y)
    f = inverse_model(epochs, *params)[:, np.newaxis]
    return np.hstack((X, f))

def add_log_feature(X, epochs):
    f = (10 * (np.log(100 * epochs - 90)))[:, np.newaxis]
    return np.hstack((X, f))

def add_optimal_log_feature(X, epochs, y):
    params, _ = fit_and_evaluate(log_model, epochs, y, p0=[1, 1, 1, 1])
    f = log_model(epochs, *params)[:, np.newaxis]
    return np.hstack((X, f))

def add_square_feature(X, epochs):
    f = 100 * (epochs ** 2 / 10_000)[:, np.newaxis]
    return np.hstack((X, f))

def add_seasonality_features(X, y, splitting_epoch, k=5, L=10):
    periods, freqs = extract_top_k_seasonalities(y[:splitting_epoch], k, L)
    print(f"Detected periods: {periods}, frequencies: {freqs}")
    seasonality_features = build_fourier_features(X, freqs)
    return np.hstack((X, seasonality_features))

def add_target_noise_feature(X, y):
    f = (y + np.random.normal(0, 0.5, size=y.shape))[:, np.newaxis]
    return np.hstack((X, f))

def add_target_as_feature(X, y):
    f = y[:, np.newaxis]
    return np.hstack((X, f))

def plot_learning_curve(epochs, y, X_test, y_pred, splitting_epoch):
    """
    Utility function to plot the learning curve along with any engineered features and predictions.

    Plots the actual dev accuracy, any engineered features, and the predicted dev accuracy on the same graph.

    Args:
        epochs (np.ndarray): Array of epoch numbers.
        y (np.ndarray): Accuracy values.
        X_test (np.ndarray): Test features used for prediction.
        y_pred (np.ndarray): Predicted dev accuracy values.
        splitting_epoch (int): Epoch at which the train/test split occurs.

    """
    # plot everything
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, y, 'o-', label='Actual Dev Accuracy')

    # plot any engineered features we added
    if X.shape[1] > 1:
        for i in range(1, X.shape[1]):
            plt.plot(epochs, X[:, i], 'x--', label=f'Feature {i}', alpha=0.5)

    plt.plot(X_test[:, 0], y_pred, 'o--', label='Predicted Dev Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Dev Accuracy')
    plt.title('Learning Curve')
    plt.axvline(x=splitting_epoch, color='red', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    args = parser.parse_args()
    splitting_epoch = args.splitting_epoch
    
    # pull learning curve data from the log file
    learning_curve_data = extract_epoch_dev_accuracy(args.log_file_path, acc_in_percent=True)

    # set up the base dataset and extract epochs once
    epochs = learning_curve_data[:, 0]
    X = epochs.reshape(-1, 1) 
    y = learning_curve_data[:, 1]                 

    # mix and match features here
    # X = add_trig_features(X, epochs)
    # X = add_inverse_feature(X, epochs)
    # X = add_optimal_inverse_feature(X, epochs, y)
    # X = add_log_feature(X, epochs)
    # X = add_optimal_log_feature(X, epochs, y)
    # X = add_square_feature(X, epochs)
    # X = add_seasonality_features(X, y, splitting_epoch, k=5, L=10)
    # X = add_target_noise_feature(X, y)
    # X = add_target_as_feature(X, y)

    # split into train and test sets
    X_train, X_test = X[:splitting_epoch], X[splitting_epoch:]
    y_train, y_test = y[:splitting_epoch], y[splitting_epoch:]

    # train the model
    regressor = TabPFNRegressor()
    regressor.fit(X_train, y_train)
    
    # predict and evaluate
    y_pred = regressor.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Mean Squared Error: {mse}")

    # plot results
    plot_learning_curve(epochs, y, X_test, y_pred, splitting_epoch)

