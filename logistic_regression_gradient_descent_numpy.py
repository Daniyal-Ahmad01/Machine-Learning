import numpy as np

# 0 = fail, 1 = pass
x = np.array([
    [2, 3],
    [1, 5],
    [6, 2],
    [7, 3],
    [8, 1],
    [3, 6],
], dtype=float)

y = np.array([0, 0, 1, 1, 1, 0], dtype=float)

# add bias term
x = np.c_[np.ones(len(x)), x]

weights = np.zeros(x.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

lr = 0.01

for epoch in range(1000):
    z = x @ weights
    p = sigmoid(z)

    # avoid log(0)
    eps = 1e-9
    loss = -np.mean(
        y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
    )

    gradient = x.T @ (p - y) / len(x)
    weights -= lr * gradient

    if epoch % 100 == 0:
        preds = (p >= 0.5).astype(int)
        print(f"Epoch {epoch}: loss={loss:.4f}, preds={preds}, weights={weights}")