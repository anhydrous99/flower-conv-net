import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_model(model, path):
    model.save(path)


def plot_history(history, plot_path):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.savefig(plot_path)


def parse_textfile(path):
    image_paths_list = []
    classifications_list = []
    with open(path, 'r') as fp:
        line = fp.readline()
        i = 0
        while line:
            s = line.split()
            image_paths_list.append(s[0])
            classifications_list.append(int(s[1]))
            line = fp.readline()
            i = i + 1
    return image_paths_list, np.asarray(classifications_list, dtype=np.int)


def load_images(image_list):
    output_list = []
    for image_path in image_list:
        img = Image.open(image_path)
        img = img.resize((25, 25))
        img_np = np.asarray(img).reshape((img.size[1], img.size[0], 3))
        output_list.append(img_np)
    return np.stack(output_list)


def split(image_array, classifications, percent_valid, shuffle=True):
    if shuffle:
        combined = np.c_[image_array.reshape(len(image_array), -1), classifications.reshape(len(classifications), -1)]
        np.random.shuffle(combined)
        image_array_shuffled = combined[:, :image_array.size // len(image_array)].reshape(image_array.shape)
        classifications = combined[:, image_array.size // len(image_array):].reshape(classifications.shape)
        image_array = image_array_shuffled

    N = image_array.shape[0] // percent_valid
    x_test = image_array[:N, :, :, :]
    y_test = classifications[:N]
    x_train = image_array[N:, :, :, :]
    y_train = classifications[N:]
    return x_train, y_train, x_test, y_test
