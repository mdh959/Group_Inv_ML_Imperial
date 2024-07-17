
from keras import layers, models
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PrepworkSasakian import daattavya_accuracy,data_wrangle_S

def permute_vector(vector):
    return np.random.permutation(vector)

def train_network(X_train, y_train, X_test, y_test):
    model = get_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    history = model.fit(
        X_train, y_train,
        epochs=999999,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return model, history

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
   
    # apply equivariant layers
    e1 = equivariant_layer(inp, 1, number_of_channels)
    e1 = layers.Dropout(0.5)(e1)
  
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    e2 = layers.Dropout(0.5)(e2)
    
    # pooling function
    p1 = layers.GlobalAveragePooling1D()(e2)

    # further training
    fc1 = layers.Dense(16, activation='relu')(p1)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    fc3 = layers.Dense(16, activation='relu')(fc2)

    out = layers.Dense(1, activation='linear')(fc3)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer = tf.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

# Running the program:

if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing
    accuracy = []
    num_runs = 10
    for _ in range(num_runs):
      # Train network on permuted data
      model, history = train_network(X_train, y_train, X_test, y_test)
      accuracy.append(daattavya_accuracy(y_train, X_test, y_test, model))
    print(accuracy)