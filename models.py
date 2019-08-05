from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def CNN(input_shape,
        num_classes=10,
        num_conv=2,
        num_dense=2,
        activation='relu',
        pooling=True,
        batch_norm=False,
        dropout=0.0,
        units=32,
        filters=32,
        kernel_size=3,
        pool_size=2,
        padding='same'
        ):

    model = Sequential()
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation,
                     padding=padding, input_shape=input_shape))

    for _ in range(num_conv-1):
        model.add(Conv2D(filters=filters, kernel_size=kernel_size,
                         activation=activation, padding=padding))

        if pooling:
            model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Dropout(dropout))

    model.add(Flatten())

    for _ in range(num_dense-1):
        model.add(Dense(units=units, activation=activation))
        model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax'))
    return model


def DNN(input_shape,
        num_classes=10,
        num_dense=4,
        units=32,
        activation='relu',
        dropout=0.0
        ):

    model = Sequential()

    for _ in range(num_dense-1):
        model.add(Dense(units=units, activation=activation))
        model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax'))
    return model