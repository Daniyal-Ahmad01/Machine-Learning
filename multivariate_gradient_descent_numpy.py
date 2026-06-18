import numpy as np

features = np.array([
    [10, 11],
    [12, 2],
    [4, 2]
])

weights = np.array([0.01, 0.01])

actual = np.array([10, 5, 3])

lr = 0.01

for epoch in range(50):
    total_error = 0

    for i in range(len(features)):
        pred = np.dot(features[i], weights)
        error = pred - actual[i]
        total_error += error
        weights = weights - lr * error * features[i]

    mean_error = total_error / len(features)
    print(f"epoch {epoch}: mean_error={mean_error:.4f}")