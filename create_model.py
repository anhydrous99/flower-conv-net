import math
import numpy as np
import tensorflow as tf
import horovod.keras as hvd
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, \
    Dropout, BatchNormalization, Activation, Input, concatenate
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import to_categorical

from utils import plot_history


def conv_block(x, nb_filter, nb_row, nb_col, padding='same', strides=(1, 1), use_bias=False):
    x = Conv2D(nb_filter, (nb_row, nb_col), strides=strides, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = Activation("relu")(x)
    return x


def stem(input):
    x = conv_block(input, 32, 3, 3, strides=(2, 2), padding='same')
    x = conv_block(x, 32, 3, 3, padding="same")
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='same')

    x = concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding='same')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding='same')

    x = concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='same')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x1, x2], axis=-1)
    return x


def inception_A(input):
    a1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    a1 = conv_block(a1, 96, 1, 1)

    a2 = conv_block(input, 96, 1, 1)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)

    merged = concatenate([a1, a2, a3, a4], axis=-1)
    return merged


def inception_B(input):
    b1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    b1 = conv_block(b1, 128, 1, 1)

    b2 = conv_block(input, 384, 1, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 256, 7, 1)

    b4 = conv_block(input, 192, 1, 1)
    b4 = conv_block(b4, 192, 7, 1)
    b4 = conv_block(b4, 224, 1, 7)
    b4 = conv_block(b4, 224, 7, 1)
    b4 = conv_block(b4, 256, 1, 7)

    merged = concatenate([b1, b2, b3, b4], axis=-1)
    return merged


def inception_C(input):
    c1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    c1 = conv_block(c1, 256, 1, 1)

    c2 = conv_block(input, 256, 1, 1)

    c3 = conv_block(input, 364, 1, 1)
    c31 = conv_block(c3, 256, 1, 3)
    c32 = conv_block(c3, 256, 3, 1)

    c4 = conv_block(input, 384, 1, 1)
    c4 = conv_block(c4, 448, 1, 3)
    c4 = conv_block(c4, 512, 3, 1)
    c41 = conv_block(c4, 256, 3, 1)
    c42 = conv_block(c4, 256, 1, 3)

    merged = concatenate([c1, c2, c31, c32, c41, c42], axis=-1)
    return merged


def reduction_A(input, k=192, l=224, m=256, n=384):
    ra1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)

    ra2 = conv_block(input, n, 3, 3, strides=(2, 2), padding='same')

    ra3 = conv_block(input, k, 1, 1)
    ra3 = conv_block(ra3, l, 3, 3)
    ra3 = conv_block(ra3, m, 3, 3, strides=(2, 2), padding='same')

    merged = concatenate([ra1, ra2, ra3], axis=-1)
    return merged


def reduction_B(input):
    rb1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)

    rb2 = conv_block(input, 192, 1, 1)
    rb2 = conv_block(rb2, 192, 3, 3, strides=(2, 2), padding='same')

    rb3 = conv_block(input, 256, 1, 1)
    rb3 = conv_block(rb3, 256, 1, 7)
    rb3 = conv_block(rb3, 320, 7, 1)
    rb3 = conv_block(rb3, 320, 3, 3, strides=(2, 2), padding='same')

    merged = concatenate([rb1, rb2, rb3], axis=-1)
    return merged


def create_model(x_train, y_train, x_test, y_test, batch_size, epochs, learning_rate,
                 plot_path='', use_data_aug=True):
    # Horovod: initialize Horobod.
    hvd.init()

    #Horovod: pin Threads to be used to process local ran
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.keras.backend.set_session(tf.Session(config=config))

    # Horovod: adjust number of epochs based on number of threads
    epochs = int(math.ceil(12.0 / hvd.size()))

    # Calculate number of classes
    n_classes = max(np.amax(y_train), np.amax(y_test)) + 1

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)
    print(y_train.shape)

    # Convert images from integers to floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Set the image pixel range of [0, 255] to [-1, 1]
    x_train = 2 * x_train / 255 - 1
    x_test = 2 * x_test / 255 - 1

    # Subtract pixel mean
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_test -= mean

    # Create Model
    init = Input(shape=(299, 299, 3))
    model = stem(init)
    for i in range(4):
        model = inception_A(model)

    model = reduction_A(model, k=192, l=224, m=256, n=384)

    for i in range(7):
        model = inception_B(model)

    model = reduction_B(model)

    for i in range(3):
        model = inception_C(model)

    model = AveragePooling2D((8, 8))(model)

    model = Dropout(0.2)(model)
    model = Flatten()(model)
    model = Dense(units=n_classes, activation='softmax')(model)
    model = Model(init, model, name='Inception-v4')

    # Initiate a Stochastic Gradient Descent optimizer
    optimizer = SGD(lr=learning_rate * hvd.size())

    # Horovod: add Horovod Distributed Optimizer.
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Training Options
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initializaion of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    # Train the model
    if use_data_aug:
        data_generator = ImageDataGenerator(width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        data_generator.fit(x_train)
        flow = data_generator.flow(x_train, y_train, batch_size=batch_size)
        history = model.fit_generator(flow,
                                      epochs=epochs,
                                      validation_data=(x_test, y_test),
                                      shuffle=True)
    else:
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True)

    # Create Plot
    if hvd.rank() == 0:
        if plot_path:
            plot_history(history, plot_path)

    return model
