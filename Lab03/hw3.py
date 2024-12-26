import time

import numpy as np
from matplotlib import pyplot as plt
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
hidden_layers = 1
hidden_size = 100
output_size = 10
learning_rate = 0.005

weights = []
biases = []


def weight_and_bias_initialization(activation_type):
    global weights, biases
    weights.clear()
    biases.clear()

    if activation_type:
        limit_input = np.sqrt(1.0 / input_size)
        weights.append(
            np.random.uniform(-limit_input, limit_input, (input_size, hidden_size))
        )
        biases.append(np.zeros((1, hidden_size)))

        for _ in range(hidden_layers - 1):
            limit_hidden = np.sqrt(1.0 / hidden_size)
            weights.append(
                np.random.uniform(
                    -limit_hidden, limit_hidden, (hidden_size, hidden_size)
                )
            )
            biases.append(np.zeros((1, hidden_size)))

        limit_output = np.sqrt(1.0 / hidden_size)
        weights.append(
            np.random.uniform(-limit_output, limit_output, (hidden_size, output_size))
        )
        biases.append(np.zeros((1, output_size)))

    else:
        limit_input = np.sqrt(2.0 / input_size)
        weights.append(np.random.randn(input_size, hidden_size) * limit_input)
        biases.append(np.zeros((1, hidden_size)))

        for _ in range(hidden_layers - 1):
            limit_hidden = np.sqrt(2.0 / hidden_size)
            weights.append(np.random.randn(hidden_size, hidden_size) * limit_hidden)
            biases.append(np.zeros((1, hidden_size)))

        limit_output = np.sqrt(2.0 / hidden_size)
        weights.append(np.random.randn(hidden_size, output_size) * limit_output)
        biases.append(np.zeros((1, output_size)))


def apply_dropout(layer, dropout_rate):
    mask = (np.random.rand(*layer.shape) > dropout_rate).astype(float)
    return layer * mask / (1 - dropout_rate)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward_propagation(x, activation_type, dropout_rate=0.3, apply_dropout_flag=False):
    activations = [x]

    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = sigmoid(z) if activation_type else relu(z)

        if apply_dropout_flag:
            a = apply_dropout(a, dropout_rate)

        activations.append(a)

    z_output = np.dot(activations[-1], weights[-1]) + biases[-1]
    a_output = softmax(z_output)
    activations.append(a_output)

    return activations


def cross_entropy_loss(predictions, targets):
    predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))


def backpropagation(activations, y, activation_type):
    global weights, biases
    layer_errors = [activations[-1] - y]  # ∂L/∂zL (ultimul layer) aka δL

    for i in range(len(weights) - 2, -1, -1):
        if activation_type:
            error = sigmoid_derivative(activations[i + 1]) * np.dot(
                layer_errors[-1], weights[i + 1].T
            )
        else:
            error = relu_derivative(activations[i + 1]) * np.dot(
                layer_errors[-1], weights[i + 1].T
            )

        layer_errors.append(error)

    layer_errors.reverse()

    # ∂L/∂w_ij = ∂L/z * dz/∂w_ij

    for i in range(len(weights)):
        # ∂L/∂w_ij = a_i × δⱼ
        gradient_w = np.dot(activations[i].T, layer_errors[i]) / y.shape[0]
        gradient_b = np.sum(layer_errors[i], axis=0, keepdims=True) / y.shape[0]
        weights[i] -= learning_rate * gradient_w
        biases[i] -= learning_rate * gradient_b


def plot_history(training_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(training_history["train_loss"], label="Train Loss")
    ax1.plot(training_history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(training_history["train_acc"], label="Train Accuracy")
    ax2.plot(training_history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Accuracy History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def predictions_visualizer(
    x_val,
    y_val,
    num_samples,
    activation_type=1,
    dropout_rate=0.3,
    apply_dropout_flag=False,
):
    indices = np.random.choice(len(x_val), num_samples, replace=False)
    samples = x_val[indices]
    true_labels = np.argmax(y_val[indices], axis=1)

    activations = forward_propagation(
        samples,
        activation_type,
        dropout_rate=dropout_rate,
        apply_dropout_flag=apply_dropout_flag,
    )
    predicted_labels = np.argmax(activations[-1], axis=1)

    print("\nSample Predictions:")
    print("True Label | Predicted Label | Correct?")
    print("-" * 40)

    for i, (true_label, pred_label) in enumerate(zip(true_labels, predicted_labels)):
        correct = "✓" if true_label == pred_label else "✗"
        print(f"{true_label:^10} | {pred_label:^14} | {correct:^8}")

        if true_label != pred_label:
            print(f"Misclassified image index: {indices[i]}")
            plt.imshow(samples[i].reshape(28, 28), cmap="gray")
            plt.title(f"True: {true_label}, Predicted: {pred_label}")
            plt.show()


def train(
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=10,
    initial_lr=0.1,
    batch_size=64,
    decay_factor=0.5,
    epochs_wh_improvement=5,
    activation_type=1,
    dropout_rate=0.3,
    apply_dropout_flag=False,
):

    accuracy_met = True
    print(f"Activation function used: " + "Sigmoid" if activation_type else "Relu")
    if apply_dropout_flag:
        print(f"Dropout will be used with a rate of: {dropout_rate}")
    print(
        f"Learning Rate Scheduler is set to 5 epoches without an improvement and a rate of decay of 0.5"
    )

    global learning_rate
    learning_rate = initial_lr
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    weight_and_bias_initialization(activation_type)

    start_time = time.time()

    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            activations = forward_propagation(
                x_batch,
                activation_type,
                dropout_rate=dropout_rate,
                apply_dropout_flag=apply_dropout_flag,
            )
            backpropagation(activations, y_batch, activation_type)

        #################################

        activations_val = forward_propagation(x_val, activation_type)
        val_loss = cross_entropy_loss(activations_val[-1], y_val)

        val_predictions = np.argmax(activations_val[-1], axis=1)
        val_labels = np.argmax(y_val, axis=1)
        accuracy = np.mean(val_predictions == val_labels)

        #################################

        activations_train = forward_propagation(x_train, activation_type)
        train_loss = cross_entropy_loss(activations_train[-1], y_train)

        train_predictions = np.argmax(activations_train[-1], axis=1)
        train_labels = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(train_predictions == train_labels)

        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["train_acc"].append(train_accuracy)
        training_history["val_acc"].append(accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs}, Learning Rate: {learning_rate:.4f}, Best Val Loss: {best_val_loss:.4f}, Val "
            f"Loss: {val_loss:.4f}, Train Loss: {train_loss:.4f}, Validation Accuracy: {accuracy:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= epochs_wh_improvement:
            learning_rate *= decay_factor
            print(f"Learning rate decay: {learning_rate:.6f}")
            epochs_since_improvement = 0

        if learning_rate < 0.000001:
            break

        if accuracy > 0.95 and accuracy_met:
            accuracy_met = False
            print(
                f"Accuracy of at least 95% was achieved after: {time.time() - start_time:.2f} seconds"
            )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training time: {total_time:.2f} seconds")
    plot_history(training_history)
    predictions_visualizer(
        x_val,
        y_val,
        num_samples=100,
        activation_type=1,
        dropout_rate=0.3,
        apply_dropout_flag=True,
    )


train(
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=500,
    initial_lr=0.1,
    batch_size=64,
    decay_factor=0.5,
    epochs_wh_improvement=5,
    activation_type=1,
    dropout_rate=0.3,
    apply_dropout_flag=True,
)
