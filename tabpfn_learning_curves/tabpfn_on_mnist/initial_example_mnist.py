import numpy as np
import matplotlib.pyplot as plt
import argparse
from contextlib import redirect_stdout

from tabpfn_learning_curves.utils.mnist_pipeline import load_learning_curve_data, split_data_compute_predictions_and_metrics, plot_learning_curve, modify_plotting_parameters, SCALING_FACTOR
from tabpfn_learning_curves.utils.top_k_seasonalities import extract_top_k_seasonalities, build_fourier_features
from tabpfn_learning_curves.utils.find_function_opt_params import inverse_model, log_model, fit_and_evaluate

parser = argparse.ArgumentParser(description="Predict learning curves using TabPFN")
parser.add_argument('--splitting_epoch', type=int, default=20, help='Epoch at which to split train and test data')
parser.add_argument('--log_file_path', type=str, default="tabpfn_learning_curves/MNIST_example_runs/at=mlp,bs=256,cs=,d=cosine,e=100,hls=32,lr=0.001,lrf=0.0005,m=0.005,o=SGD,s=42,i=2/logs.txt", help='Path to the log file containing learning curve data')
parser.add_argument('--path_to_save_plot', type=str, default=None, help='Path to save the plot of the learning curve, if not provided, the plot will be shown instead')
parser.add_argument('--add_trig_features', nargs="*", choices=['sin_2pi', 'cos_2pi', 'sin_400', 'cos_400'], default=[], help='Whether to add trigonometric features based on the epoch number, choose from: sin_2pi, cos_2pi, sin_400, cos_400')
parser.add_argument('--add_inverse_feature', default=False, action='store_true', help='Whether to add the inverse feature 1 - 1/epoch')
parser.add_argument('--add_optimal_inverse_feature', default=False, action='store_true', help='Whether to add an optimally parameterized inverse function as a feature')
parser.add_argument('--add_log_feature', default=False, action='store_true', help='Whether to add a logarithmic feature')
parser.add_argument('--add_optimal_log_feature', default=False, action='store_true', help='Whether to add an optimally parameterized logarithmic function as a feature')
parser.add_argument('--add_square_feature', default=False, action='store_true', help='Whether to add the square of the epoch as a feature')
parser.add_argument('--add_seasonality_features', type=int, default=0, help='Whether to add seasonality features based on top k detected seasonalities in the training data (provide k)')
parser.add_argument('--add_target_noise_feature', default=False, action='store_true', help='Whether to add a feature that is the target value with some noise added')
parser.add_argument('--add_target_as_feature', default=False, action='store_true', help='Whether to add the target value itself as a feature')
parser.add_argument('--run_multiple_added_features', default=False, action='store_true', help='Whether to run multiple experiments with different combinations of added features and cumulate the plots')
parser.add_argument('--output_text_file', type=str, default=None, help='Path to save the textual output of the experiment results (MSE values), if not provided, results will be printed to console')

# feature engineering functions

def add_trig_features(X, epochs, which_trig_features=[], feature_names=None):
    features_to_add = []
    if 'sin_2pi' in which_trig_features:
        features_to_add.append((SCALING_FACTOR * np.sin(epochs)[:, np.newaxis], 'sinus s periodou 2π'))
    if 'cos_2pi' in which_trig_features:
        features_to_add.append((SCALING_FACTOR * np.cos(epochs)[:, np.newaxis], 'cosinus s periodou 2π'))
    if 'sin_400' in which_trig_features:
        features_to_add.append((SCALING_FACTOR * np.sin(epochs * 2 * np.pi / 400)[:, np.newaxis], 'sinus s periodou 400'))
    if 'cos_400' in which_trig_features:
        features_to_add.append((SCALING_FACTOR * np.cos(epochs * 2 * np.pi / 400)[:, np.newaxis], 'cosinus s periodou 400'))

    for feature, name in features_to_add:
        X = np.hstack((X, feature))
        if feature_names is not None:
            feature_names.append(name)

    return X

def add_inverse_feature(X, epochs, feature_names=None):
    f = SCALING_FACTOR * (1 - 1 / (epochs))[:, np.newaxis]
    if feature_names is not None:
        feature_names.extend(['rostoucí asymptotická funkce'])
    return np.hstack((X, f))

def add_optimal_inverse_feature(X, epochs, y, feature_names=None):
    params, _ = fit_and_evaluate(inverse_model, epochs, y)
    f = inverse_model(epochs, *params)[:, np.newaxis]
    if feature_names is not None:
        feature_names.extend(['optimalizovaná asymptotická funkce'])
    return np.hstack((X, f))

def add_log_feature(X, epochs, feature_names=None):
    f = (10 * (np.log(SCALING_FACTOR * epochs - 90)))[:, np.newaxis]
    if feature_names is not None:
        feature_names.extend(['logaritmus'])
    return np.hstack((X, f))

def add_optimal_log_feature(X, epochs, y, feature_names=None):
    params, _ = fit_and_evaluate(log_model, epochs, y, p0=[1, 1, 1, 1])
    f = log_model(epochs, *params)[:, np.newaxis]
    if feature_names is not None:
        feature_names.extend(['optimalizovaný logaritmus'])
    return np.hstack((X, f))

def add_square_feature(X, epochs, feature_names=None):
    f = SCALING_FACTOR * (epochs ** 2 / 10_000)[:, np.newaxis]
    if feature_names is not None:
        feature_names.extend(['čtverec epochy'])
    return np.hstack((X, f))

def add_seasonality_features(X, y, splitting_epoch, k=5, L=10, feature_names=None):
    periods, freqs = extract_top_k_seasonalities(y[:splitting_epoch], k, L)
    print(f"Detected periods: {periods}, frequencies: {freqs}")
    seasonality_features = build_fourier_features(X[:, 0], freqs) * SCALING_FACTOR
    if feature_names is not None:
        for period in periods:
            feature_names.append(f'sezónnost, sinus s periodou {period}')
            feature_names.append(f'sezónnost, cosinus s periodou {period}')
    return np.hstack((X, seasonality_features))

def add_target_noise_feature(X, y, seed=42, feature_names=None):
    np.random.seed(seed)
    f = (y + np.random.normal(0, 0.5, size=y.shape))[:, np.newaxis]
    if feature_names is not None:
        feature_names.extend(['zašuměná cílová hodnota'])
    return np.hstack((X, f))

def add_target_as_feature(X, y, feature_names=None):
    f = y[:, np.newaxis]
    if feature_names is not None:
        feature_names.extend(['cílová hodnota'])
    return np.hstack((X, f))

def compute_one_experiment(args: argparse.Namespace):
    splitting_epoch = args.splitting_epoch
    
    # Load learning curve data
    X, y = load_learning_curve_data(args.log_file_path)
    epochs = X[:, 0]  # Extract epoch numbers for feature engineering

    # feature names for plotting
    feature_names = []

    # add features based on command line arguments
    X = add_trig_features(X, epochs, which_trig_features=args.add_trig_features, feature_names=feature_names)
    if args.add_inverse_feature:
        X = add_inverse_feature(X, epochs, feature_names)
    if args.add_optimal_inverse_feature:
        X = add_optimal_inverse_feature(X, epochs, y, feature_names)
    if args.add_log_feature:
        X = add_log_feature(X, epochs, feature_names)
    if args.add_optimal_log_feature:
        X = add_optimal_log_feature(X, epochs, y, feature_names)
    if args.add_square_feature:
        X = add_square_feature(X, epochs, feature_names)
    if args.add_seasonality_features != 0:
        X = add_seasonality_features(X, y, splitting_epoch, k=args.add_seasonality_features, L=10, feature_names=feature_names)
    if args.add_target_noise_feature:
        X = add_target_noise_feature(X, y, feature_names=feature_names)
    if args.add_target_as_feature:
        X = add_target_as_feature(X, y, feature_names)

    # split the data, compute predictions and metrics
    y_pred, mse, last_pred_squared_error = split_data_compute_predictions_and_metrics(X, y, splitting_epoch)
    print(f"Mean Squared Error: {mse}")
    print(f"Squared error of the last prediction: {last_pred_squared_error}")

    return epochs, X, y, y_pred, splitting_epoch, mse, feature_names

def prepare_features_for_multiple_experiments(args):
    features_list = []
    for feature in args.add_trig_features:
        features_list.append(('add_trig_features', [feature]))
    setattr(args, 'add_trig_features', [])  # initialize with empty list
    if args.add_inverse_feature:
        features_list.append(('add_inverse_feature', True))
    setattr(args, 'add_inverse_feature', False)  # initialize with False
    if args.add_optimal_inverse_feature:
        features_list.append(('add_optimal_inverse_feature', True))
    setattr(args, 'add_optimal_inverse_feature', False)  # initialize with False
    if args.add_log_feature:
        features_list.append(('add_log_feature', True))
    setattr(args, 'add_log_feature', False)  # initialize with False
    if args.add_optimal_log_feature:
        features_list.append(('add_optimal_log_feature', True))
    setattr(args, 'add_optimal_log_feature', False)  # initialize with False
    if args.add_square_feature:
        features_list.append(('add_square_feature', True))
    setattr(args, 'add_square_feature', False)  # initialize with False
    if args.add_seasonality_features != 0:
        for i in range(args.add_seasonality_features):
            features_list.append(('add_seasonality_features', i + 1))  # we will loop over different values of k for seasonality features
    setattr(args, 'add_seasonality_features', 0)  # initialize with 0
    if args.add_target_noise_feature:
        features_list.append(('add_target_noise_feature', True))
    setattr(args, 'add_target_noise_feature', False)  # initialize with False
    if args.add_target_as_feature:
        features_list.append(('add_target_as_feature', True))
    setattr(args, 'add_target_as_feature', False)  # initialize with False

    return features_list


def run_multiple_added_features(args: argparse.Namespace):
    # set up nicer plot style
    modify_plotting_parameters()

    features_list = prepare_features_for_multiple_experiments(args)

    if len(features_list) == 0:
        print("No features to add, running a single experiment with just the epoch number as a feature.")
        epochs, X, y, y_pred, splitting_epoch, mse, feature_names = compute_one_experiment(args)
        plot_learning_curve(epochs, X, y, y_pred, splitting_epoch, mse=mse, feature_names=feature_names, path_to_save=args.path_to_save_plot)
        return
    
    
    height = (len(features_list) + 1) // 2  # number of rows needed for subplots
    fig, axs = plt.subplots(height, 2, figsize=(10, height * 3))
    axs = axs.flatten()

    for i, (feature, value) in enumerate(features_list):
        print(f"Running experiment with added feature: {feature} - {value}")
        previous_value = getattr(args, feature)  # store previous value to reset later
        setattr(args, feature, value)
        epochs, X, y, y_pred, splitting_epoch, mse, feature_names = compute_one_experiment(args)
        plot_learning_curve(epochs, X, y, y_pred, splitting_epoch, ax=axs[i], path_to_save=None)
        if feature != 'add_seasonality_features':
            axs[i].set_title(f'Příznak: {feature_names[0]}')
        else:
            axs[i].set_title(f'Příznak: {value} sezónní příznak{"y" if value > 1 else ""}')
        if i % 2 == 0:
            axs[i].set_ylabel('Přesnost (%)')
        if i  == len(axs) - 1:
            axs[i].legend()
        setattr(args, feature, previous_value)  # reset the feature flag for the next experiment
    
    plt.tight_layout()
    if args.path_to_save_plot:
        plt.savefig(args.path_to_save_plot, dpi=300)
    else:
        plt.show()

def execute_experiments(args):
    """Handles the core logic for running experiments."""
    if args.run_multiple_added_features:
        run_multiple_added_features(args)
    else:
        # For a single experiment with specific features added based on command line args
        epochs, X, y, y_pred, splitting_epoch, mse, feature_names = compute_one_experiment(args)
        plot_learning_curve(
            epochs, X, y, y_pred, splitting_epoch, 
            mse=mse, feature_names=feature_names, path_to_save=args.path_to_save_plot
        )

if __name__ == "__main__":
    args = parser.parse_args()

    if args.output_text_file:
        with open(args.output_text_file, "w", encoding="utf-8") as f, redirect_stdout(f):
            execute_experiments(args)
    else:
        execute_experiments(args)

