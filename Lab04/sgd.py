import numpy as np
from torchvision.datasets import MNIST
from sklearn.utils import shuffle
import time

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(),
                download=True,
                train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return mnist_data, mnist_labels

def normalize(data):
    data = np.array(data, dtype=np.float32) / 255.0
    return data

def encode(labels, num_classes=10):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X=normalize(train_X)
test_X=normalize(test_X)

train_Y_encoded = encode(train_Y)
test_Y_encoded = encode(test_Y)


input_size = 784
hidden_layer_size = 100
output_size = 10
learning_rate = 0.001

weights_hid = np.random.randn(input_size, hidden_layer_size) * np.sqrt(1/ input_size)
weights_out = np.random.randn(hidden_layer_size, output_size) * np.sqrt(1/ hidden_layer_size)
bias_hid = np.zeros((1,hidden_layer_size))
bias_out = np.zeros((1,output_size))

reduce_after=5
best_acc=0
no_improvement=0

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

def derivate_sigmoid(x):
    return x * (1 - x)

def softmax(z):  #?
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(preds, labels):
    return -np.mean(np.sum(labels * np.log(preds + 1e-8), axis=1))

def forward(x):
    z_hid = np.dot(x, weights_hid) + bias_hid
    a_hid = sigmoid(z_hid)
    z_out = np.dot(a_hid, weights_out, ) + bias_out
    a_out = softmax(z_out)

    return a_hid, a_out

def backpropagation(x, y, a_hid, a_out):
    global weights_out, weights_hid, bias_out, bias_hid
    error_out = a_out - y
    gradient_out = np.dot(a_hid.T, error_out) #?

    error_hid = np.dot(error_out, weights_out.T) * derivate_sigmoid(a_hid)
    gradient_hid = np.dot(x.T, error_hid)

    weights_out = weights_out - learning_rate * gradient_out
    bias_out = bias_out - learning_rate * np.sum(error_out, axis=0, keepdims=True)
    weights_hid = weights_hid - learning_rate * gradient_hid
    bias_hid -= learning_rate * np.sum(error_hid, axis=0, keepdims=True)



def train(x_train, y_train, x_test, y_test, epochs, batch_size):

    global learning_rate, best_acc, no_improvement
    accuracy_met = True
    start_time = time.time()
    for epoch in range(epochs):
        x_train,y_train = shuffle(x_train, y_train)

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            a_hid, a_out = forward(x_batch)
            backpropagation(x_batch, y_batch, a_hid, a_out)

        a_hid_train, a_out_train = forward(x_train)
        a_hid_test, a_out_test = forward(x_test)

        train_loss = cross_entropy(a_out_train, y_train)
        test_loss = cross_entropy(a_out_test, y_test)

        predictions = np.argmax(a_out_test, axis=1)
        labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == labels)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy * 100:.4f}')

        if accuracy > best_acc:
            best_acc = accuracy
            no_improvement=0
        else: no_improvement += 1

        if no_improvement >= reduce_after:
            learning_rate *= 0.5
            no_improvement = 0

        if accuracy > 0.95 and accuracy_met:
            accuracy_met = False
            print(f"95% accuracy was found in {time.time() - start_time:.2f} seconds")

    print(f"{time.time() - start_time:.2f} seconds")


train(train_X, train_Y_encoded, test_X, test_Y_encoded, epochs=150, batch_size=50)