import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


def get_mlp_model(input_size, hidden_layer_size, dropout_factor, output_size):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_factor))
    model.add(Dense(hidden_layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_factor))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_cnn_model(input_shape, filters, kernel_size, pool_size, dropout_factor, output_size):
    model = Sequential()
    model.add(Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu'))
    model.add(Flatten())
    model.add(Dropout(dropout_factor))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def run_comparison():
    # load data from mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # count unique labels. In case of MNIST, ten elements, labels from 0 to 9
    output_size = len(np.unique(y_train))

    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    image_size = x_train.shape[1]

    # normalize pixel values to range 0.0 .. 1.0
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    print("==== Multilayer perceptron ====")
    # MLP hyperparameters
    batch_size_mlp = 128
    input_size_mlp = image_size * image_size
    hidden_layer_size_mlp = 256
    dropout_factor_mlp = 0.45

    # resize input to single dimensional vector
    x_train = np.reshape(x_train, [-1, input_size_mlp])
    x_test = np.reshape(x_test, [-1, input_size_mlp])

    # get and train MLP
    model_mlp = get_mlp_model(input_size_mlp, hidden_layer_size_mlp, dropout_factor_mlp, output_size)
    start_mlp = time.time()
    model_mlp.fit(x_train, y_train, epochs=20, batch_size=batch_size_mlp)
    end_mlp = time.time()
    training_time_mlp = end_mlp - start_mlp

    # evaluate MLP on test data
    _, accuracy_mlp = model_mlp.evaluate(x_test, y_test, batch_size=batch_size_mlp, verbose=0)

    print("==== Convolution Neural Network ====")
    # CNN hyperparameters
    batch_size_cnn = 128
    input_shape = (image_size, image_size, 1)
    kernel_size_cnn = 3
    pool_size_cnn = 2
    filters_cnn = 64
    dropout_factor_cnn = 0.2

    # resize input to two dimensional image with one channel
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

    # get and train CNN
    model_cnn = get_cnn_model(input_shape, filters_cnn, kernel_size_cnn, pool_size_cnn, dropout_factor_cnn, output_size)
    start_cnn = time.time()
    model_cnn.fit(x_train, y_train, epochs=10, batch_size=batch_size_cnn)
    end_cnn = time.time()
    training_time_cnn = end_cnn - start_cnn

    # evaluate cnn on test data
    _, accuracy_cnn = model_cnn.evaluate(x_test, y_test, batch_size=batch_size_cnn, verbose=0)

    # print results
    print("\nMLP accuracy: %.1f%%. Training time: %.1fs. Parameters count: %d. Batch count: %d" %
          (100.0 * accuracy_mlp, training_time_mlp, model_mlp.count_params(), batch_size_mlp))
    print("\nCNN accuracy: %.1f%%. Training time: %.1fs. Parameters count: %d. Batch count: %d" %
          (100.0 * accuracy_cnn, training_time_cnn, model_cnn.count_params(), batch_size_cnn))


run_comparison()
