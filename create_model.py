
import numpy as np
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical


def create_model(x_train, y_train, x_test, y_test, n_classes, batch_size, epochs,
                 plot_path='', use_data_aug=True):
    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

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
    model = Sequential()
    model.add(Conv2D(32,                           # Number of learnable filters
                     (5, 5),                       # Size of filters
                     data_format='channels_last',  # Format of data (#images, height, width, channels)
                     activation='relu',            # Kind of activation layer
                     input_shape=(25, 25, 3)))     # Shape of input, channels last
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    # Training Options
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Train the model
    if use_data_aug:
        data_generator = ImageDataGenerator(width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            horizontal_flip=True)
        data_generator.fit(x_train)
        history = model.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch_size),
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
    if plot_path:
        plot_history(history, plot_path)  # TODO

    return model
