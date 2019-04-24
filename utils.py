import matplotlib.pyplot as plt
import numpy as np


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
            s = line.split('	')
            image_paths_list[i] = s[0]
            classifications_list[i] = int(s[1])
            line = fp.readline()
            i = i + 1
    return image_paths_list, np.asarray(classifications_list, dtype=np.int)
