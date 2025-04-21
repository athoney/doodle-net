import numpy as np

# 1. create network architecture
L = 3
n = [2, 3, 3, 1]

# 2. create weights and biases
W1 = np.random.randn(n[1],n[0])
W2 = np.random.randn(n[2],n[1])
W3 = np.random.randn(n[3],n[2])

b1 = np.random.randn(1, n[1])
b2 = np.random.randn(1, n[2])
b3 = np.random.randn(1, n[3])

# 3. create training data and labels
def prepare_data():
    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])

    y = np.array([
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0
    ])
    return X, y

# 4. create activation function
def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

# 5. create feed forward process
def feed_forward(A0):
    m = A0.shape[0]
    print(A0.shape)
    
    #layer 1
    Z1 = A0 @ W1.T + b1
    assert Z1.shape == (m, n[1])
    A1 = sigmoid(Z1)

    #layer 2
    Z2 = A1 @ W2.T + b2
    A2 = sigmoid(Z2)

    #layer 3
    Z3 = A2 @ W3.T + b3
    A3 = sigmoid(Z3)
    return A3


A0, y = prepare_data()
y_hat = feed_forward(A0)
print(y_hat)

#compute cost
def cost(y_hat, y):
    #m * n^L
    losses = - ((y.T* np.log(y_hat.T)) + (1-y.T)*np.log(1-y_hat.T))

    m = y_hat.reshape(-1).shape[0]

    summed_losses = (1/m) * np.sum(losses, axis=1)

    # only necessary when output layer > 1 node
    return np.sum(summed_losses)

print(cost(y_hat, y))


















