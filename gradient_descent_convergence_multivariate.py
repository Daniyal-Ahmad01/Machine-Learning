import numpy as np

d = np.array([
    [5, 8],
    [4, 9],
    [8, 5],
    [8, 2]
])

weights = np.array([0.0, 0.0])
actual = np.array([100, 56, 35, 98])

lr = 0.01

prev_loss = float('inf')

for epoch in range(5000):
    total_error = 0

    for i in range(len(d)):
        pred = np.dot(weights, d[i])
        error = pred - actual[i]
        weights = weights - lr * error * d[i]
        total_error += error ** 2

    mse = total_error / len(d)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: MSE={mse:.4f}, Weights={weights}")

    if abs(prev_loss - mse) < 0.001:
        print("Converged at epoch:", epoch)
        break

    prev_loss = mse