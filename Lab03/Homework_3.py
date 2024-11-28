from torchvision.datasets import MNIST
import numpy as np


def download_mnist(is_train: bool):
    dataset = MNIST(
        root="./data",
        transform=lambda x: np.array(x).flatten(),
        download=True,
        train=is_train,
    )

    mnist_data = []
    mnist_labels = []

    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return np.array(mnist_data), np.array(mnist_labels)


def normalize_data(data):
    return data.astype(np.float32) / 255.0


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def initialize_parameters(input_size=784, hidden_size=100, output_size=10):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    return W1, b1, W2, b2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    exp_logits = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]


def forward_propagation(X, weights, biases):
    hidden_layer_input = np.dot(X, weights["W1"]) + biases["b1"]
    hidden_layer_output = relu(hidden_layer_input)

    return np.dot(hidden_layer_output, weights["W2"]) + biases["b2"]


def backpropagation(X, y, weights, biases, learning_rate):

    y_pred = forward_propagation()

    # Calculate loss
    loss = mean_squared_error(y, y_pred)

    # Backpropagation
    # Output layer gradients
    d_loss = mean_squared_error_derivative(y, y_pred)
    dW2 = np.dot(hidden_layer_output.T, d_loss)
    db2 = np.sum(d_loss, axis=0, keepdims=True)

    # Hidden layer gradients
    d_hidden_layer = np.dot(d_loss, weights["W2"].T) * relu_derivative(
        hidden_layer_input
    )
    dW1 = np.dot(X.T, d_hidden_layer)
    db1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Update weights and biases
    weights["W1"] -= learning_rate * dW1
    biases["b1"] -= learning_rate * db1
    weights["W2"] -= learning_rate * dW2
    biases["b2"] -= learning_rate * db2

    return loss


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = normalize_data(train_X)
train_Y = one_hot_encode(train_Y)

test_X = normalize_data(test_X)

input_size = 784
hidden_size = 100
output_size = 10
learning_rate = 0.01
num_epochs = 10
batch_size = 64

W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
weights = {"W1": W1, "W2": W2}
biases = {"b1": b1, "b2": b2}

for epoch in range(num_epochs):
    for i in range(0, train_X.shape[0], batch_size):
        X_batch = train_X[i : i + batch_size]
        y_batch = train_Y[i : i + batch_size]

        loss = backpropagation(X_batch, y_batch, weights, biases, learning_rate)

    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Note: After training, you will want to evaluate the model on the test set.
