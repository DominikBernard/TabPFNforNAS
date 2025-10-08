import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os

from tabpfn import TabPFNRegressor

from log_runs_to_tensors import extract_logs_from_subdirectories, create_tensors_from_logs

splitting_epoch = 20  # Define the epoch at which to split training and testing data
rotation = 9  # Define how many times to rotate the list of runs

# get the data from the log files with current directory as root
# current derictory is path to this file without the filename
learning_curve_data_list = create_tensors_from_logs(extract_logs_from_subdirectories(os.path.dirname(__file__)))

# rotate the data
for _ in range(rotation):
    learning_curve_data_list.append(learning_curve_data_list.pop(0))

# Prepare dataset
X_list = []
y_list = []
for i in range(len(learning_curve_data_list)):
    learning_curve_data = learning_curve_data_list[i]
    X_i = learning_curve_data[:, 0].reshape(-1, 1)  # Epoch numbers as features
    y_i = learning_curve_data[:, 1]  # Dev accuracy as target values

    # add i as a first feature to distinguish runs
    X_i = np.hstack((np.full((X_i.shape[0], 1), i), X_i))

    X_list.append(X_i)
    y_list.append(y_i)

# split the data into training and testing sets
# last run is split into train and test
X_train = np.vstack(X_list[:-1] + [X_list[-1][:splitting_epoch]])
X_test = X_list[-1][splitting_epoch:]

y_train = np.hstack(y_list[:-1] + [y_list[-1][:splitting_epoch]])
y_test = y_list[-1][splitting_epoch:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Initialize a regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# measure the mean squared error
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# Plot the results for visualization
plt.figure(figsize=(10, 6))
# plot training runs
for i in range(len(learning_curve_data_list) - 1):
    plt.plot(learning_curve_data_list[i][:, 0], learning_curve_data_list[i][:, 1], '--', label=f'Run {i+1} (Train)', alpha=0.5)

# plot test run
plt.plot(learning_curve_data_list[-1][:, 0], learning_curve_data_list[-1][:, 1], 'g--', label=f'Run {len(learning_curve_data_list)} (Test)')

plt.plot(X_test[:, 1], y_pred, 'r.-', label='Predicted Dev Accuracy (Test)')
plt.axvline(x=splitting_epoch, color='k', linestyle='--', label='Train/Test Split')
plt.xlabel('Epoch')
plt.ylabel('Dev Accuracy')
plt.title('Dev Accuracy vs Epoch')
plt.legend()

plt.grid()
# plt.show()
plt.savefig(f"plots/mnist_mlp_multiple_runs_tabpfn_example{rotation}.png")
plt.close()

