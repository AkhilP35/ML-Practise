import numpy as np

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)
y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

rng = np.random.default_rng(seed=0)
W1 = rng.normal(scale=0.5, size=(2, 2))
b1 = np.zeros((1, 2))
W2 = rng.normal(scale=0.5, size=(2, 1))
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

lr = 0.5
for epoch in range(5000):
    # forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    # loss gradient (MSE)
    error = y_hat - y

    # backprop
    dz2 = error * sigmoid_prime(z2)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    dz1 = dz2 @ W2.T * sigmoid_prime(z1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # update weights
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print(np.round(y_hat, 3))
