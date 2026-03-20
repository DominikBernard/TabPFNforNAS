# in this code, we will compute differences betweeen each point in the learning curve and the previous one
# this will give us the improvement or deterioration at each epoch
# we will then use these differences as the target values for the TabPFNRegressor


import numpy as np
import matplotlib.pyplot as plt

from tabpfn import TabPFNRegressor

from tabpfn_predicting_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy

splitting_epoch = 20  # Define the epoch at which to split training and testing data

# Extract learning curve data from log file
log_file_path = "tabpfn_predicting_learning_curves/logs/run1.txt"
learning_curve_data = extract_epoch_dev_accuracy(log_file_path)

# Prepare dataset
X = learning_curve_data[1:, 0].reshape(-1, 1)  # Epoch numbers as features
y = np.diff(learning_curve_data[:, 1]) # Differences in accuracy as target values

# # add sine and cosine features to X
# X = np.hstack((X, np.sin(X[:, 0])[:, np.newaxis]))
# X = np.hstack((X, np.cos(X[:, 0])[:, np.newaxis]))
# X = np.hstack((X, np.sin(X[:, 0] * np.pi / 200)[:, np.newaxis]))
# X = np.hstack((X, np.cos(X[:, 0] * np.pi / 200)[:, np.newaxis]))

# # add 1-1/x feature to X
# X = np.hstack((X, (0.9 - 1 / (X[:, 0] + 1))[:, np.newaxis]))

# # add log feature to X
# X = np.hstack((X, ((np.log(X[:, 0] + 1))/3)[:, np.newaxis]))

# # add square feature to X
# X = np.hstack((X, (X[:, 0] ** 2 / 10_000)[:, np.newaxis]))

# # add ultimate feature
# X = np.hstack((X, (np.sin(X[:, 0] * np.pi / 200) + 2 * (1 - 3 / (X[:, 0] + 1)) / 3)[:, np.newaxis]))

# add ulitimate feature number 2
# it is a combination feature 4 until step 9 and feature 3 after step 10
# X = np.hstack((X, (np.where(X[:, 0] < 10, ((np.log(X[:, 0] + 1))/3), (0.9 - 1 / (X[:, 0] + 1))))[:, np.newaxis]))

# add ultimate feature number 3
# it is a target feature, with some noise
# X = np.hstack((X, (y + np.random.normal(0, 0.005, size=y.shape))[:, np.newaxis]))

# add target as a feature
# X = np.hstack((X, y[:, np.newaxis]))

# split the data into training and testing sets
X_train, X_test = X[:splitting_epoch], X[splitting_epoch:]
y_train, y_test = y[:splitting_epoch], y[splitting_epoch:]


# Initialize a regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)
# Make predictions
y_pred = regressor.predict(X_test)

# measure the mean squared error
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# Plot the results for visualization
# with a straight line splitting training and testing data
plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], y, 'o-', label='Actual Dev Accuracy')

if X.shape[1] > 1:
    for i in range(1, X.shape[1]):
        plt.plot(X[:, 0], X[:, i], 'x--', label=f'Feature {i}', alpha=0.5)

plt.plot(X_test[:, 0], y_pred, 'o--', label='Predicted Dev Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Dev Accuracy')
plt.title('Learning Curve')
plt.axvline(x=splitting_epoch, color='red', linestyle='--', label='Train/Test Split')
plt.legend()
plt.grid()
plt.show()

