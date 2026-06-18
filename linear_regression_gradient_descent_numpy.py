import numpy as np

x = np.array([10, 20, 30, 40])
y = np.array([50, 100, 150, 200])

weight = 0.1
lr = 0.001

for step in range(50):
    pred = weight * x
    error = pred - y
    gradient = np.mean(error * x)
    weight = weight - lr * gradient

    if step % 10 == 0:
        print(f"step {step}: weight={weight:.4f}, error_mean={np.mean(error):.4f}")