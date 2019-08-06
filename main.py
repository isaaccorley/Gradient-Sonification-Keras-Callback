import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from itertools import product

from models import CNN, DNN
from callbacks import GradientSonification


path = 'files/'

if not os.path.exists(path):
    os.mkdir(path)

batch_size = 32
num_classes = 10
epochs = 1

fs = 44100
duration = 0.01
freq = 200.0

# Load MNIST
(X_train, y_train), _ = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_train /= 255
y_train = to_categorical(y_train, num_classes)

# Param space
learning_rates = [0.1, 0.01,]
optimizers = [SGD, Adam, RMSprop, Adadelta]
activations = ['tanh']

for act, lr, opt in product(activations, learning_rates, optimizers):

    # Train CNN
    fname = '_'.join(['cnn', opt.__name__.lower(), str(lr), act])
    fname = path + fname + '.wav'
    print(fname)
    model = CNN(input_shape=(28,28,1),
                activation=act)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt(lr=lr),
                  metrics=['accuracy'])

    grad_son = GradientSonification(path=fname,
                                    model=model,
                                    fs=fs,
                                    duration=duration,
                                    freq=freq)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt(lr=lr),
                  metrics=['accuracy'] + grad_son.metrics)


    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[grad_son])


    # Train DNN
    fname = '_'.join(['dnn', opt.__name__.lower(), str(lr), act])
    fname = path + fname + '.wav'
    print(fname)
    model = DNN(input_shape=(28*28,),
                activation=act)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt(lr=lr),
                  metrics=['accuracy'])

    grad_son = GradientSonification(path=fname,
                                    model=model,
                                    fs=fs,
                                    duration=duration,
                                    freq=freq)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt(lr=lr),
                  metrics=['accuracy'] + grad_son.metrics)

    model.fit(X_train.reshape(X_train.shape[0], 28*28, 1).squeeze(), y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[grad_son])
