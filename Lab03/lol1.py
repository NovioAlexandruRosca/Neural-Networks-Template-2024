import numpy as np
from torchvision.datasets import MNIST
from sklearn.utils import shuffle


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


x_train, y_train = download_mnist(is_train=True)
x_val, y_val = download_mnist(is_train=False)

x_train = normalize_data(x_train)
x_val = normalize_data(x_val)

y_train = one_hot_encode(y_train)
y_val = one_hot_encode(y_val)


input_size = 784
hidden_size = 100
output_size = 10
learning_rate = 0.01

weights_input_hidden = np.random.randn(input_size, hidden_size) * (
    1.0 / np.sqrt(input_size)
)
weights_hidden_output = np.random.randn(hidden_size, output_size) * (
    1.0 / np.sqrt(input_size)
)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))


def l2_regularization(weights, lambda_l2=0.001):
    l2_penalty = 0
    for w in weights:
        l2_penalty += np.sum(w**2)
    return lambda_l2 * l2_penalty


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward_propagation(x, dropout_rate=0.5):
    z_hidden = np.dot(x, weights_input_hidden) + bias_hidden
    a_hidden = sigmoid(z_hidden)

    # if dropout_rate > 0:
    #     mask = np.random.binomial(1, 1 - dropout_rate, size=a_hidden.shape)
    #     a_hidden *= mask

    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = softmax(z_output)

    return a_hidden, a_output


def cross_entropy_loss(predictions, targets, weights, lambda_l2=0.01):
    predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    # l2_penalty = l2_regularization(weights, lambda_l2)
    l2_penalty = 0
    cross_entropy = -np.mean(np.sum(targets * np.log(predictions), axis=1))
    return cross_entropy + l2_penalty


def backpropagation(x, y, a_hidden, a_output, lambda_l2=0.01):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    error_output = a_output - y
    gradient_output = np.dot(a_hidden.T, error_output)

    error_hidden = np.dot(error_output, weights_hidden_output.T) * sigmoid_derivative(
        a_hidden
    )
    gradient_hidden = np.dot(x.T, error_hidden)

    weights_hidden_output -= learning_rate * (
        gradient_output + lambda_l2 * weights_hidden_output
    )
    bias_output -= learning_rate * np.sum(error_output, axis=0, keepdims=True)

    weights_input_hidden -= learning_rate * (
        gradient_hidden + lambda_l2 * weights_input_hidden
    )
    bias_hidden -= learning_rate * np.sum(error_hidden, axis=0, keepdims=True)


def train(
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=10,
    batch_size=64,
    initial_lr=0.01,
    decay_factor=0.5,
    patience=5,
    dropout_rate=0.5,
):

    global learning_rate
    learning_rate = initial_lr
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            a_hidden, a_output = forward_propagation(x_batch, dropout_rate=dropout_rate)
            backpropagation(x_batch, y_batch, a_hidden, a_output)

        _, a_output_train = forward_propagation(x_train, dropout_rate=0)
        _, a_output_val = forward_propagation(x_val, dropout_rate=0)
        train_loss = cross_entropy_loss(
            a_output_train, y_train, [weights_input_hidden, weights_hidden_output]
        )
        val_loss = cross_entropy_loss(
            a_output_val, y_val, [weights_input_hidden, weights_hidden_output]
        )

        val_predictions = np.argmax(a_output_val, axis=1)
        val_labels = np.argmax(y_val, axis=1)
        accuracy = np.mean(val_predictions == val_labels)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            learning_rate *= decay_factor
            print(f"Learning rate decayed to: {learning_rate:.6f}")
            epochs_since_improvement = 0


train(x_train, y_train, x_val, y_val, epochs=100, batch_size=64)
