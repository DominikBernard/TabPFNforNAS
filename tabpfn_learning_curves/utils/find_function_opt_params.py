import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
from typing import Optional

def inverse_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Model a - (b/x)."""
    return a - (b / x)

def log_model(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """The logarithmic model: a * log(bx + c) + d."""
    # We suppress warnings here just in case the optimizer probes a negative value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a * np.log(b * x + c) + d

def fit_and_evaluate(model_func: callable, x_data: np.ndarray, y_data: np.ndarray, p0: Optional[list] = None, bounds: tuple = (-np.inf, np.inf)):
    """
    Generic function to fit a model to the data and calculate MSE.
    
    Args:
        model_func: The function to fit (e.g., inverse_model, log_model, ...)
        x_data, y_data: The datasets
        p0: Initial guesses for the parameters
        bounds: Bounds for the parameters

    Returns:
        popt: Optimal parameters for the model
        mse: Mean Squared Error of the fit
    """
    # Fit the curve.
    popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
    
    # Calculate Mean Squared Error
    predictions = model_func(x_data, *popt)
    mse = np.mean((y_data - predictions) ** 2)
    
    return popt, mse

if __name__ == "__main__":

    def plot_results(x_data: np.ndarray, y_data: np.ndarray, model_func: callable, params: list, mse: float, label: str):
        """Helper function to plot the results."""
        plt.plot(x_data, y_data, label='Actual Data', color='black', marker='x')
        x_smooth = np.linspace(min(x_data), max(x_data), 100)
        plt.plot(x_smooth, model_func(x_smooth, *params), 
                 label=f'{label} (MSE: {mse:.4f})', linestyle='--')
        plt.title('Model Fit to Learning Curve')
        plt.xlabel('Epoch (x)')
        plt.ylabel('Score (y)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # data

    from tabpfn_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy
    # Extract learning curve data from log file
    log_file_path = "tabpfn_learning_curves/MNIST_example_runs/at=cnn,bs=1024,cs=conv3x3-BN-relu-conv3x3-BN-relu-linear,d=linear,e=100,hls=32,lr=0.0005,lrf=1e-05,m=0.005,o=SGD,s=42,i=9/logs.txt"
    learning_curve_data = extract_epoch_dev_accuracy(log_file_path, acc_in_percent=True)
    epochs = learning_curve_data[:, 0]
    scores = learning_curve_data[:, 1]

    # --- Fit Model 1: Inverse ---
    params_inv, mse_inv = fit_and_evaluate(inverse_model, epochs, scores)
    print("--- Inverse Model: y = a - (b/x) ---")
    print(f"Optimal Parameters: a={params_inv[0]:.4f}, b={params_inv[1]:.4f}")
    print(f"MSE: {mse_inv}\n")

    # --- Fit Model 2: Logarithmic ---
    params_log, mse_log = fit_and_evaluate(log_model, epochs, scores, p0=[1, 1, 1, 1])
    print("--- Logarithmic Model: y = a * ln(bx + c) + d ---")
    print(f"Optimal Parameters: a={params_log[0]:.4f}, b={params_log[1]:.4f}, c={params_log[2]:.4f}, d={params_log[3]:.4f}")
    print(f"MSE: {mse_log}\n")

    # plot results for both models
    plot_results(epochs, scores, inverse_model, params_inv, mse_inv, label='Inverse Model')
    plot_results(epochs, scores, log_model, params_log, mse_log, label='Logarithmic Model')
