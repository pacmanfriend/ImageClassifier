from keras.datasets import mnist
import numpy as np
from neural import Network, load_image_cv
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)


def main():
    gui = GUI()
    gui.load_mnist_data()
    gui.start()


def train():
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
    # model.init_weights()
    # model.fit(x_train=images, y_train=labels, batch_size=32, epochs=50, alpha=0.1)
    model.load_model('models/mnist_1_model.hdf5')
    model.evaluate(x_test=test_images, y_test=test_labels)
    # model.save_model('models/mnist_1_model.hdf5')


class GUI:
    def __init__(self):
        self.__root = Tk()

        self.grayscale_image: np.ndarray = None
        self.model: Network = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.result = StringVar()
        self.threads_count = IntVar()

        self.buttons_frame = Frame(self.__root, width=50, pady=10, padx=10)
        self.load_img_btn = Button(self.buttons_frame, text="Загрузить изображение", padx=2, pady=2, width=30, height=1,
                                   bg='white', fg='black')

        self.create_model_btn = Button(self.buttons_frame, text="Создать модель", padx=2, pady=2,
                                       width=30, height=1, bg='white', fg='black')
        self.train_model_btn = Button(self.buttons_frame, text="Обучить модель", padx=2, pady=2,
                                      width=30, height=1, bg='white', fg='black')
        self.save_model_btn = Button(self.buttons_frame, text="Сохранить модель", padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black')

        self.load_model_btn = Button(self.buttons_frame, text='Загрузить модель', padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black')
        self.predict_btn = Button(self.buttons_frame, text='Определить цифру', padx=2, pady=2,
                                  width=30, height=1, bg='white', fg='black')

        self.show_train_btn = Button(self.buttons_frame, text="Тренировочные данные", padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black')

        self.image_plot = None
        self.canvas = None

    def start(self):
        self.__root.title("Image Classifier")
        self.__root.geometry("1280x720+50+50")

        self.load_img_btn.bind("<Button-1>", self.load_image_from_filesystem)
        self.load_model_btn.bind('<Button-1>', self.load_model)
        self.predict_btn.bind('<Button-1>', self.get_result)
        self.show_train_btn.bind('<Button-1>', self.show_mnist)

        self.buttons_frame.pack(anchor=NW)
        self.load_img_btn.pack(anchor=NW)
        self.load_model_btn.pack(anchor=NW)
        self.predict_btn.pack(anchor=NW)
        self.show_train_btn.pack(anchor=NW)

        fig = plt.figure(figsize=(6, 4))
        self.image_plot = fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(fig, master=self.__root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(anchor=NE)

        self.__root.mainloop()

    def load_model(self, event):
        model_path = filedialog.askopenfilename()

        if model_path != "":
            self.model = Network()
            self.model.load_model(model_path)

    def get_result(self, event):
        res = self.model.predict(self.grayscale_image)

        a = 0

    def load_image_from_filesystem(self, event):
        filepath = filedialog.askopenfilename()

        if filepath != "":
            self.grayscale_image = load_image_cv(filepath)

            self.image_plot.imshow(self.grayscale_image)

            self.canvas.draw()
            self.canvas.get_tk_widget().pack(anchor=NE)

            # toolbar = NavigationToolbar2Tk(canvas, root)
            # toolbar.update()
            # canvas.get_tk_widget().pack(anchor=NE)

    def load_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def show_mnist(self, event):
        mnist_win = Tk()
        mnist_win.geometry('1280x720+50+50')

        fig = plt.figure(figsize=(10, 10))
        for i in range(36):
            image_plot = fig.add_subplot(6, 6, i + 1)
            # image_plot.xticks([])
            # image_plot.yticks([])
            image_plot.grid(False)
            image_plot.imshow(self.x_train[i].reshape((28, 28)))

        canvas = FigureCanvasTkAgg(fig, master=mnist_win)
        canvas.draw()
        canvas.get_tk_widget().pack(anchor=CENTER)


if __name__ == '__main__':
    main()
