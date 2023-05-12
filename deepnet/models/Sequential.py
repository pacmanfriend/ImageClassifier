from deepnet.layers import (Input, Output, Relu, Softmax, Sigmoid, Dense)


class Sequential:
    def __init__(self):
        self.input_layer = Input(shape=(28 * 28, 1))
        self.relu = Relu()
        self.softmax = Softmax()

    def fit(self, x_train, y_train, batch_size, epochs, validation_split=0.1):
        pass

    def __update_weights(self):
        pass

    def predict(self, input_data):
        pass

    def evaluate(self, x_test, y_test):
        pass

    def copy(self):
        pass

    def compile(self):
        pass
