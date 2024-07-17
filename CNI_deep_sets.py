from keras import layers, models
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from prepwork import data_wrangle_CNI, classification_accuracy
from PrepworkSasakian import train_network

def permute_vector(vector):
    return np.random.permutation(vector)

# Define custom layer for expand_dims and tile
class ExpandTileLayer(layers.Layer):
    def call(self, inputs):
        out2 = tf.expand_dims(inputs, axis=1)
        out2 = tf.tile(out2, [1, 5, 1])
        return out2

# Define new architecture for the NN
def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):
    # ---(1)---
    out1 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=False, activation='relu')(inp)
    # ---(2)---
    out2 = layers.GlobalAveragePooling1D()(inp)
    out2 = ExpandTileLayer()(out2)
    out2 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=True, activation='relu')(out2)
    return layers.Add()([out1, out2])

def get_network():
    number_of_channels = 100
    inp = layers.Input(shape=(5, 1))
   
    # Apply equivariant layers
    e1 = equivariant_layer(inp, 1, number_of_channels)
    e1 = layers.Dropout(0.5)(e1)
  
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    e2 = layers.Dropout(0.5)(e2)
    
    # Pooling function
    p1 = layers.GlobalAveragePooling1D()(e2)

    # Further training
    fc1 = layers.Dense(16, activation='relu')(p1)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    fc3 = layers.Dense(16, activation='relu')(fc2)

    out = tf.keras.layers.Dense(24, activation='softmax')(fc3)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

# Running the program
if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_CNI()
    y = (y-1)/2
    y = np.round(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing
    accuracy = []
    num_runs = 5
    for _ in range(num_runs):
        # Train network on permuted data
        model, history = train_network(X_train, y_train, X_test, y_test, get_network())
        accuracy.append(classification_accuracy(X_test, y_test, model))
    print(accuracy)
