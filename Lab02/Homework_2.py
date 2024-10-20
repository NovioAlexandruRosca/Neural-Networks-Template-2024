import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

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


def initialize_parameters(input_size=784, output_size=10):
    w = np.random.randn(input_size, output_size) * 0.01
    b = np.zeros((1, output_size))
    return w, b


def softmax(z):
    exp_logits = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def forward_pass(X, W, b):
    z = np.dot(X, W) + b
    y_hat = softmax(z)
    return y_hat


def compute_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def train(X, y, num_epochs=10, batch_size=100, learning_rate=0.01):
    num_features = X.shape[1]
    num_classes = y.shape[1]
    W, b = initialize_parameters(num_features, num_classes)

    num_samples = X.shape[0]

    for epoch in range(num_epochs):
        epoch_loss = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_X = X[start:end]
            batch_y = y[start:end]

            y_hat = forward_pass(batch_X, W, b)

            batch_loss = compute_loss(batch_y, y_hat)
            epoch_loss += batch_loss

            error = batch_y - y_hat

            W += learning_rate * np.dot(batch_X.T, error) / batch_size
            b += learning_rate * np.sum(error, axis=0, keepdims=True) / batch_size

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / (num_samples / batch_size):.4f}")

    return W, b


def predict(X, W, b):
    y_hat = forward_pass(X, W, b)
    return np.argmax(y_hat, axis=1)


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = normalize_data(train_X)
train_Y = one_hot_encode(train_Y)

test_X = normalize_data(test_X)
test_Y = one_hot_encode(test_Y)

num_classes = 10
W, b = train(train_X, train_Y, num_epochs=1, batch_size=100, learning_rate=0.5)

predictions = predict(test_X, W, b)


def calculate_accuracy(predictions, test_Y):
    predictions_length = predictions.shape[0]
    correct_pred = 0

    for prediction, true_label in zip(predictions, test_Y):
        true_class = np.argmax(true_label)

        if prediction == true_class:
            correct_pred += 1

    return correct_pred / predictions_length * 100


accuracy = calculate_accuracy(predictions, test_Y)
print(f"Accuracy: {accuracy:.2f}%")