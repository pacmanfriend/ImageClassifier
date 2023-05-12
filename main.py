from keras.datasets import mnist
import numpy as np
from neural import Network


def main():
    data_size = 60000

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    images, labels = (x_train[:data_size].reshape(data_size, 28 * 28) / 255, y_train[:data_size])
    one_hot_labels = np.zeros((len(labels), 10))

    for i, j in enumerate(labels):
        one_hot_labels[i][j] = 1

    labels = one_hot_labels

    test_images = x_test.reshape(len(x_test), 28 * 28) / 255
    test_labels = np.zeros((len(y_test), 10))

    for i, j in enumerate(y_test):
        test_labels[i][j] = 1

    model = Network()
    model.init_weights()
    model.fit(x_train=images, y_train=labels, batch_size=32, epochs=50, alpha=0.1)
    model.evaluate(x_test=test_images, y_test=test_labels)


if __name__ == '__main__':
    main()
