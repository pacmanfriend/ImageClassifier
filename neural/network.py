import numpy as np
import sys
import time


class Network:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 100
        self.output_size = 10

        self.weights_0_1 = None
        self.weights_1_2 = None
        self.weights_2_3 = None

    def init_weights(self):
        self.weights_0_1 = 0.2 * np.random.random((self.input_size, self.hidden_size)) - 0.1
        self.weights_1_2 = 0.2 * np.random.random((self.hidden_size, self.hidden_size)) - 0.1
        self.weights_2_3 = 0.2 * np.random.random((self.hidden_size, self.output_size)) - 0.1

    def fit(self, x_train, y_train, batch_size, epochs, validation_split=0.1, alpha=0.01):
        validation_size = int(len(x_train) * validation_split)
        train_size = int(len(x_train) - validation_size)

        for e in range(epochs):
            error, correct_cnt = (0.0, 0)
            test_error, test_correct_cnt = (0.0, 0)

            # np.random.shuffle(x_train)
            # np.random.shuffle(y_train)

            train_images = x_train[:train_size]
            train_labels = y_train[:train_size]

            validation_images = x_train[train_size:]
            validation_labels = y_train[train_size:]

            start = time.monotonic()

            for i in range(len(train_images)):
                batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

                layer_0 = train_images[batch_start:batch_end]
                layer_1 = tanh(np.dot(layer_0, self.weights_0_1))
                layer_2 = tanh(np.dot(layer_1, self.weights_1_2))
                # dropout_mask = np.random.randint(2, size=layer_1.shape)
                # layer_1 *= dropout_mask * 2
                layer_3 = softmax(np.dot(layer_2, self.weights_2_3))

                for k in range(layer_3.shape[0]):
                    correct_cnt += int(
                        np.argmax(layer_3[k:k + 1]) == np.argmax(train_labels[batch_start + k:batch_start + k + 1]))

                layer_3_delta = (train_labels[batch_start:batch_end] - layer_3) / (batch_size * layer_3.shape[0])
                layer_2_delta = layer_3_delta.dot(self.weights_2_3.T) * tanh2deriv(layer_2)
                layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * tanh2deriv(layer_1)
                # layer_1_delta *= dropout_mask

                self.weights_2_3 += alpha * layer_2.T.dot(layer_3_delta)
                self.weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
                self.weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

            for i in range(len(validation_images)):
                layer_0 = validation_images[i:i + 1]
                layer_1 = tanh(np.dot(layer_0, self.weights_0_1))
                layer_2 = tanh(np.dot(layer_1, self.weights_1_2))
                layer_3 = softmax(np.dot(layer_2, self.weights_2_3))

                test_correct_cnt += int(np.argmax(layer_3) == np.argmax(validation_labels[i:i + 1]))

            end = time.monotonic() - start

            # if j % 10 == 0:
            sys.stdout.write(
                f"I:{e} || Test-Acc:{test_correct_cnt / float(validation_size)} "
                f"|| Train-Acc:{correct_cnt / float(train_size)} "
                f"|| {end}s\n")

    def predict(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass


relu = lambda x: (x >= 0) * x
relu2deriv = lambda x: x >= 0


def tanh(x):
    return np.tanh(x)


def tanh2deriv(x):
    return 1 - (x ** 2)


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)
