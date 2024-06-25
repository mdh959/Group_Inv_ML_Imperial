#Import libraries
import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
import urllib.request

#Import weights (input features) and CY hodge (target labels)

with open('/content/WP4s.txt','r') as file:
    weights = eval(file.read())
with open('/content/WP4_Hodges (1).txt','r') as file:
    CYhodge = eval(file.read())
CY = [[weights[i],CYhodge[i]] for i in range(7555)]

#Import sasakian hodge
Sweights, SHodge = [], []
with open('/content/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx%6 == 0: Sweights.append(eval(line))
        if idx%6 == 2: SHodge.append(eval(line))
del(file,line,idx)

Sweights = np.array(Sweights)
SHodge = np.array(SHodge)

print(Sweights.shape)

# Convert to NumPy arrays
X = np.array(weights)
y = np.array(CYhodge)

def get_network():
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Reshape((5,))(inp)
    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
    h1_drop = tf.keras.layers.Dropout(0.2)(h1)  # Adding dropout after h1
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1_drop)
    h2_drop = tf.keras.layers.Dropout(0.2)(h2)  # Adding dropout after h2
    h3 = tf.keras.layers.Dense(16, activation='relu')(h2_drop)
    h3_drop = tf.keras.layers.Dropout(0.2)(h3)  # Adding dropout after h3
    out = tf.keras.layers.Dense(2, activation='linear')(h3_drop)

    model = tf.keras.models.Model(inputs=inp, outputs=out)

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

def train_network(X_train, y_train, X_test, y_test):
    model = get_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return history
if __name__ == '__main__':
    model = get_network()
    print(model.summary())
    X_train, X_test, y_train, y_test = train_test_split(Sweights, SHodge, test_size=0.5)
    print(f'Test Accuracy of Neural Network after one run: {train_network(X_train, y_train, X_test, y_test)}')
