from keras.datasets import mnist
import numpy as np
import time
from neural import Network, load_image_cv
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from multiprocessing import Process, Manager, Barrier, cpu_count, current_process


def main():
    gui = GUI()
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

        self.max_train_size = 0
        self.max_test_size = 0

        self.result = StringVar()
        self.threads_count = IntVar()

        self.load_mnist_data()

        self.buttons_frame = Frame(self.__root, width=50, pady=10, padx=10)
        self.result_frame = Frame(self.__root, width=50, padx=10, pady=10)
        self.max_params_frame = Frame(self.__root, padx=5, pady=5)
        self.train_frame = Frame(self.__root, padx=5, pady=5)
        self.threads_counter_frame = Frame(self.__root, padx=5, pady=5)

        self.load_img_btn = Button(self.buttons_frame, text="Загрузить изображение", padx=2, pady=2, width=30, height=1,
                                   bg='white', fg='black', command=self.load_image_from_filesystem)
        self.create_model_btn = Button(self.buttons_frame, text="Создать модель", padx=2, pady=2,
                                       width=30, height=1, bg='white', fg='black', command=self.create_model)
        self.train_model_btn = Button(self.threads_counter_frame, text="Начать обучение модель", padx=2, pady=2,
                                      width=30, height=1, bg='white', fg='black', command=self.train_model)
        self.save_model_btn = Button(self.buttons_frame, text="Сохранить модель", padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black', command=self.save_model)
        self.show_model_info_btn = Button(self.buttons_frame, text='Показать информацию о модели', padx=2, pady=2,
                                          width=30, height=1, bg='white', fg='black')

        self.load_model_btn = Button(self.buttons_frame, text='Загрузить модель', padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black', command=self.load_model)
        self.predict_btn = Button(self.buttons_frame, text='Определить цифру', padx=2, pady=2,
                                  width=30, height=1, bg='white', fg='black', command=self.predict)

        self.show_train_btn = Button(self.buttons_frame, text="Тренировочные данные", padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black', command=self.show_mnist)

        self.result_label = Label(self.result_frame, text="", font=("Arial", 16))

        self.max_threads_count_label = Label(self.max_params_frame,
                                             text=f"Доступное количество потоков: {cpu_count()}",
                                             font=("Arial", 10))

        self.max_train_size_label = Label(self.max_params_frame,
                                          text=f"Доступный размер обучающих данных: {len(self.x_train)}",
                                          font=("Arial", 10))

        self.train_size_label = Label(self.train_frame, text="Размер обучающих данных", font=("Arial", 10))
        self.train_size_entry = Entry(self.train_frame)

        self.threads_count_label = Label(self.threads_counter_frame, text="Количество потоков", font=("Arial", 10))
        self.threads_count_entry = Entry(self.threads_counter_frame)

        self.image_plot = None
        self.canvas = None

    def start(self):
        self.__root.title("Image Classifier")
        self.__root.geometry("1280x720+50+50")

        self.buttons_frame.pack(anchor=NW, side=LEFT)

        self.max_params_frame.pack(anchor=NW, side=TOP, expand=FALSE)
        self.max_threads_count_label.pack(anchor=NW, side=TOP)
        self.max_train_size_label.pack(anchor=NW, side=TOP)
        self.train_model_btn.pack(anchor=NW, side=BOTTOM)

        self.train_frame.pack(anchor=NW, side=TOP, expand=FALSE)
        self.train_size_label.pack(anchor=NW, side=LEFT)
        self.train_size_entry.pack(anchor=NE, side=RIGHT)

        self.threads_counter_frame.pack(anchor=NW, side=TOP, expand=FALSE)
        self.threads_count_label.pack(anchor=NW, side=LEFT)
        self.threads_count_entry.pack(anchor=NE, side=RIGHT)

        self.load_img_btn.pack(anchor=NW)
        self.load_model_btn.pack(anchor=NW)
        self.create_model_btn.pack(anchor=NW)
        self.save_model_btn.pack(anchor=NW)
        self.predict_btn.pack(anchor=NW)
        self.show_train_btn.pack(anchor=NW)

        fig = plt.figure(figsize=(6, 4))
        self.image_plot = fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(fig, master=self.__root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(anchor=NE, side=TOP)

        self.result_frame.pack(anchor=NE)
        self.result_label.pack(anchor=NE)

        self.__root.mainloop()

    def load_model(self):
        model_path = filedialog.askopenfilename()

        if model_path != "":
            self.model = Network()
            self.model.load_model(model_path)

    def predict(self):
        res = self.model.predict(self.grayscale_image)

        self.result_label['text'] = f'Результат: {res}'

    def load_image_from_filesystem(self):
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

        self.max_train_size = len(self.x_train)
        self.max_test_size = len(self.x_test)

    def show_mnist(self):
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

    def create_model(self):
        self.model = Network()
        self.model.init_weights()

    def save_model(self):
        filepath = filedialog.asksaveasfilename()

        if filepath != "":
            self.model.save_model(filepath)

    def train_model(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Загрузите или создайте модель!")
            return

        threads = int(self.threads_count_entry.get())
        train_size = int(self.train_size_entry.get())
        pack_size = train_size // threads

        print(f"Data size: {train_size}, Threads: {threads}")

        start_time = time.monotonic()

        images = self.x_train[:train_size].reshape(train_size, 28 * 28) / 255
        labels = self.y_train[:train_size]

        one_hot_labels = np.zeros((len(labels), 10))

        for i, j in enumerate(labels):
            one_hot_labels[i][j] = 1

        labels = one_hot_labels

        image_packs = []
        labels_packs = []

        for i in range(threads):
            start, end = ((i * pack_size), ((i + 1) * pack_size))
            image_packs.append(images[start:end])
            labels_packs.append(labels[start:end])

        models = []
        for _ in range(threads):
            model_copy = self.model.copy_model()
            models.append(model_copy)

        bar = Barrier(threads)
        processes = []

        start_pr = time.monotonic()

        with Manager() as m:
            grads_dict = m.dict()
            weights_dict = m.dict()

            for i in range(threads):
                pr = Process(name=f"proc-{i}", target=models[i].fit,
                             args=(
                                 image_packs[i], labels_packs[i], 16, 40, bar, grads_dict, weights_dict, 0.1, 0.1,
                                 print_proc))
                processes.append(pr)

                pr.start()

            end_pr = time.monotonic() - start_pr
            print(end_pr)

            for p in processes:
                p.join()

            for i in range(threads):
                key = f"proc-{i}"

                weights = weights_dict[key]

                models[i].weights_0_1 = weights[0]
                models[i].weights_1_2 = weights[1]
                models[i].weights_2_3 = weights[2]

        end_time = time.monotonic() - start_time
        print(end_time)

        test_images = self.x_test.reshape(len(self.x_test), 28 * 28) / 255
        test_labels = np.zeros((len(self.y_test), 10))

        for i, j in enumerate(self.y_test):
            test_labels[i][j] = 1

        evals = []
        max_ev = 0
        ind = -1
        for i in range(len(models)):
            v = models[i].evaluate(test_images, test_labels)

            if v[1] > max_ev:
                max_ev = v[1]
                ind = i

        self.model = models[ind].copy_model()

    def show_model_info(self):
        pass


def print_proc(a, b, c, d, i):
    g = current_process().name

    # print(f"{g} Epoch: {i} Train-Err: {a} Train-Acc: {b} Validation-Err: {c} Validation-Acc: {d}")


if __name__ == '__main__':
    main()
