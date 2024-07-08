from keras import layers, models, optimizers

import numpy as np
from keras import layers, models, optimizers
import numpy as np
import requests
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from PrepworkSasakian import daattavya_accuracy
from PrepworkSasakian import data_wrangle_S
from PrepworkSasakian import train_network
import urllib.request


def permute_vector(vector):
    return np.random.permutation(vector)

#define new architecture for the NN
def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):

    inp = layers.Reshape((5, number_of_channels_in))(inp)
    # ---(1)---
    out1 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=False, activation='relu')(inp)
    # ---(2)---
    out2 = layers.GlobalAveragePooling1D(pool_size=5, strides=1, padding='valid')(inp)
    repeated = [out2 for _ in range(5)]
    out2 = layers.Concatenate(axis=1)(repeated)
    out2 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=True, activation='relu')(out2)
    # return out1, out2
    return layers.Add()([out1,out2])

def get_deep_sets_network(pooling='sum'):
    number_of_channels = 10
    inp = layers.Input(shape=(5,))
    inp_list = [inp for _ in range(number_of_channels)]
    inp_duplicated = layers.Concatenate(axis=1)(inp_list)

    # Apply first equivariant layer
    e1 = equivariant_layer(inp_duplicated, number_of_channels, number_of_channels)
    e1 = layers.Dropout(0.5)(e1)
    
    # Repeat process for more layers
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    e2 = layers.Dropout(0.5)(e2)

    if pooling=='sum':
        p1 = layers.GlobalAveragePooling1D(5, strides=1, padding='valid')(e2)
    else:
        p1 = layers.MaxPooling1D(5, strides=1, padding='valid')(e2)
    p2 = layers.Flatten()(p1)

    fc1 = layers.Dense(16, activation='relu')(p2)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    fc3 = layers.Dense(16, activation='relu')(fc2)
    out = layers.Dense(1, activation='linear')(fc3)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

#running the program:

if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Split data into training and testing
    X_test_permuted = np.apply_along_axis(permute_vector, 1, X_test)
    # Train network on permuted data
    model, history = train_network(X_train, y_train, X_test, y_test)

    # Evaluate accuracy on original test set and permuted test set
    print('Accuracy as defined in the paper:')
    print(str(round(daattavya_accuracy(y_train, X_test, y_test, model) * 100, 1)) + '%')
    print(str(round(daattavya_accuracy(y_train, X_test_permuted, y_test, model) * 100, 1)) + '%')

    # Define the input vectors
    test_vectors = [
        np.array([1, 2, 3, 4, 5]),
        np.array([5, 4, 3, 2, 1])
    ]

    # Get the predictions for the input vectors
    for i, test_vector in enumerate(test_vectors, start=1):
        test_vector_reshaped = test_vector.reshape(1, -1)
        prediction = model.predict(test_vector_reshaped)
        print(f'Prediction for test vector {i}: {prediction[0][0]}')

