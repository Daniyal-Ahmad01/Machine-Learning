import numpy as np

x = np.array([
    [5, 4],
    [6, 7],
    [4, 5],
    [7, 8]
])

y = np.array([0, 1, 0, 1])

# add bias term
x = np.c_[np.ones(len(x)), x]

weights = np.zeros(x.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

lr = 0.01

for epoch in range(1000):
    z = x @ weights
    p = sigmoid(z)

    # numerical stability fix
    eps = 1e-9
    loss = -np.mean(
        y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
    )

    gradient = x.T @ (p - y) / len(x)
    weights -= lr * gradient

    if epoch % 100 == 0:
        preds = (p >= 0.5).astype(int)
        print(f"epoch {epoch}: loss={loss:.4f}, preds={preds}, weights={weights}")

print("final weights:", weights)