import numpy as np

x = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

actual = np.array([3, 7, 11, 15])

x = np.c_[np.ones(len(x)), x]  # add bias term

weights = np.zeros(x.shape[1])
lr = 0.01

for epoch in range(1000):
    pred = x @ weights
    error = pred - actual
    mse = np.mean(error ** 2)

    gradient = x.T @ error
    weights -= lr * gradient / len(x)

    if epoch % 100 == 0:
        print(f"epoch {epoch}: mse={mse:.6f}, weights={weights}")