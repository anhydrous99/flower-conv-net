import matplotlib.pyplot as plt
import pandas as pd
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


def parse_textfile():
    train_dataset = pd.read_csv('FlowerData/6k_img_map.txt', delimiter='\t', header=None, dtype=str)
    test_dataset = pd.read_csv('FlowerData/val_map.txt', delimiter='\t', header=None, dtype=str)
    return train_dataset, test_dataset


def load_images(path, image_list):
    output_list = []
    for image_path in image_list:
        img = Image.open(path + image_path)
        img = img.resize((299, 299))
        img_np = np.asarray(img).reshape((img.size[1], img.size[0], 3))
        output_list.append(img_np)
    return np.stack(output_list)