import numpy as np
from torchvision.datasets import MNIST
import time


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


def initialize_parameters(input_size=784, output_size=10):
    w = np.random.randn(input_size, output_size) * 0.05
    b = np.zeros((1, output_size))
    return w, b


def softmax(z):
    exp_logits = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def forward_prop(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)


def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def train(X, y, num_epochs=10, batch_size=100, learning_rate=0.01):
    num_features = X.shape[1]
    num_classes = y.shape[1]
    W, b = initialize_parameters(num_features, num_classes)
    print(W.shape, b.shape, X.shape, y.shape)

    num_samples = X.shape[0]

    for epoch in range(num_epochs):
        epoch_loss = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_X = X[start:end]
            batch_y = y[start:end]

            y_hat = forward_prop(batch_X, W, b)
            error = batch_y - y_hat

            batch_loss = compute_loss(batch_y, y_hat)
            epoch_loss += batch_loss

            W += learning_rate * np.dot(batch_X.T, error) / batch_size
            b += learning_rate * np.sum(error, axis=0, keepdims=True) / batch_size

        print(
            f"Epoch {epoch + 1}/{num_epochs}; EpochLoss: {epoch_loss / (num_samples / batch_size):.3f}"
        )

    return W, b


def predict(X, W, b):
    y_hat = forward_prop(X, W, b)
    return np.argmax(y_hat, axis=1)


start_download_time = time.time()

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

end_download_time = time.time()

train_X = normalize_data(train_X)
train_Y = one_hot_encode(train_Y)

test_X = normalize_data(test_X)

num_classes = 10

start_train_time = time.time()

W, b = train(train_X, train_Y, num_epochs=50, batch_size=100, learning_rate=0.1)

end_train_time = time.time()

predictions = predict(test_X, W, b)


def accuracy(predictions, test_Y):
    predictions_length = predictions.shape[0]

    # correct_pred = sum(1 for prediction, true_label in zip(predictions, test_Y) if prediction == true_label)
    correct_pred = np.sum(predictions == test_Y)

    return correct_pred / predictions_length * 100


accuracy = accuracy(predictions, test_Y)
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Download time: {end_download_time - start_download_time:.2f} sec")
print(f"Train time: {end_train_time - start_train_time:.2f} sec")
