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
    out2 = layers.GlobalAveragePooling1D()(out1)  # Correct usage of GlobalAveragePooling1D
    
    out2 = tf.expand_dims(out2, axis=1)
    out2 = tf.tile(out2, [1, 5, 1])
    out2 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=True, activation='relu')(out2)
    return layers.Add()([out1, out2])

def get_network():
    number_of_channels = 10
    inp = layers.Input(shape=(5, 1))
    
    # Apply equivariant layers
    e1 = equivariant_layer(inp, 1, number_of_channels)
    e1 = layers.Dropout(0.5)(e1)
  
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    e2 = layers.Dropout(0.5)(e2)
    
    # Pooling function
    p1 = layers.GlobalAveragePooling1D()(e2)

    fc1 = layers.Dense(16, activation='relu')(p1)
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

# Function to calculate accuracy
def daattavya_accuracy(training_outputs, test_inputs, test_outputs, model):
    bound = 0.05 * (np.max(training_outputs) - np.min(training_outputs))
    predictions = model.predict(test_inputs)
    return np.mean(np.where(np.abs(np.array(predictions) - test_outputs) < bound, 1, 0))

# Running the program:

if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_S()
    X = X.reshape(-1, 5, 1)  # Reshape the data to fit the model input shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing
    X_test_permuted = np.apply_along_axis(permute_vector, 1, X_test)
    # Train network on permuted data
    model, history = train_network(X_train, y_train, X_test, y_test)

    # Evaluate accuracy on original test set and permuted test set
    print('Accuracy as defined in the paper:')
    print(str(round(daattavya_accuracy(y_train, X_test, y_test, model) * 100, 1)) + '%')
    print(str(round(daattavya_accuracy(y_train, X_test_permuted, y_test, model) * 100, 1)) + '%')


