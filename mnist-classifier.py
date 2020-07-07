import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
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


def run_comparison():
    # load data from mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_size = x_train.shape[1]
    input_size = img_size * img_size

    # count unique labels
    output_size = len(np.unique(y_train))

    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    x_train = np.reshape(x_train, [-1, input_size])
    x_train = x_train.astype('float32') / 255
    x_test = np.reshape(x_test, [-1, input_size])
    x_test = x_test.astype('float32') / 255

    batch_size = 128
    hidden_layer_size = 256
    dropout_factor = 0.45

    model_mlp = get_mlp_model(input_size, hidden_layer_size, dropout_factor, output_size)
    model_mlp.fit(x_train, y_train, epochs=20, batch_size=batch_size)

    _, accuracy = model_mlp.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print("\nMLP network accuracy: %.1f%%" % (100.0 * accuracy))


run_comparison()
