import math
import keras
import tensorflow as tf
import horovod.keras as hvd
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, \
    Dropout, BatchNormalization, Activation, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import SGD

from utils import plot_history, load_images


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


def dataframe_max(dataframe, col_index):
    max = 0
    for i in range(len(dataframe.index)):
        if (int(dataframe.iloc[i, col_index]) > max):
            max = int(dataframe.iloc[i, col_index])
    return max


def shard(arr, shard_index, n_shards):
    shard_size = arr.shape[0] // n_shards
    shard_start = shard_index * shard_size
    shard_end = (shard_index + 1) * shard_size
    if shard_end > arr.shape[0]:
        shard_end = arr.shape[0]
    return arr[shard_start:shard_end]


def create_model(train_dataset, test_dataset, batch_size, epochs, learning_rate, plot_path=''):
    # Horovod: initialize Horobod.
    hvd.init()

    # Horovod: pin Threads to be used to process local ran
    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    keras.backend.set_session(tf.Session(config=config))

    # Horovod: adjust number of epochs based on number of threads
    epochs = int(math.ceil(epochs / hvd.size()))

    # Calculate number of classes
    n_classes = dataframe_max(train_dataset, 1) + 1

    # Shard Data based on process
    train_dataset = shard(train_dataset, hvd.rank(), hvd.size())
    test_dataset = shard(test_dataset, hvd.rank(), hvd.size())

    # create list of categories
    categories = [str(i) for i in list(range(n_classes))]

    # Create Model
    init = Input(shape=(299, 299, 3))
    model = stem(init)
    for i in range(2):
        model = inception_A(model)

    model = reduction_A(model, k=192, l=224, m=256, n=384)

    for i in range(4):
        model = inception_B(model)

    model = reduction_B(model)

    for i in range(2):
        model = inception_C(model)

    model = AveragePooling2D((8, 8))(model)

    model = Dropout(0.2)(model)
    model = Flatten()(model)
    model = Dense(units=102, activation='softmax')(model)
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
    train_generator = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        shear_range=0.3,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1. / 255.,
        validation_split=0.2)
    test_generator = ImageDataGenerator(rescale=1. / 255.)
    train_flow = train_generator.flow_from_dataframe(
        dataframe=train_dataset,
        x_col=0,
        y_col=1,
        classes=categories,
        class_mode='categorical',
        target_size=(299, 299),
        batch_size=batch_size,
        directory='FlowerData/jpg',
        subset='training')
    valid_flow = train_generator.flow_from_dataframe(
        dataframe=train_dataset,
        x_col=0,
        y_col=1,
        classes=categories,
        class_mode='categorical',
        target_size=(299, 299),
        batch_size=batch_size,
        directory='FlowerData/jpg',
        subset='validation')
    # test_flow = test_generator.flow(test_images, test_cats, batch_size=batch_size)
    history = model.fit_generator(generator=train_flow,
                                  steps_per_epoch=(train_flow.n // train_flow.batch_size),
                                  validation_data=valid_flow,
                                  validation_steps=3 * (valid_flow.n // valid_flow.batch_size),
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=callbacks)
    # model.evaluate_generator(generator=test_flow)
    # Create Plot
    if hvd.rank() == 0:
        if plot_path:
            plot_history(history, plot_path)
    return model
