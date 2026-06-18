import numpy as np

features = np.array([4, 3, 5])
weights = np.array([7.0, 5.0, 10.0])

actual = 30
learning_rate = 0.01

for step in range(15):
    prediction = np.dot(features, weights)
    error = prediction - actual
    weights = weights - learning_rate * error * features

    print(f"step {step}: pred={prediction:.2f}, error={error:.2f}, weights={weights}")