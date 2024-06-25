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

"model_36"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_37 (InputLayer)       [(None, 5)]               0         
                                                                 
 reshape_36 (Reshape)        (None, 5)                 0         
                                                                 
 dense_116 (Dense)           (None, 16)                96        
                                                                 
 dropout (Dropout)           (None, 16)                0         
                                                                 
 dense_117 (Dense)           (None, 32)                544       
                                                                 
 dropout_1 (Dropout)         (None, 32)                0         
                                                                 
 dense_118 (Dense)           (None, 16)                528       
                                                                 
 dropout_2 (Dropout)         (None, 16)                0         
                                                                 
 dense_119 (Dense)           (None, 2)                 34        
                                                                 
=================================================================
Total params: 1202 (4.70 KB)
Trainable params: 1202 (4.70 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
Epoch 1/1000
118/118 [==============================] - 2s 5ms/step - loss: 2866.8271 - accuracy: 0.7509 - val_loss: 2008.5646 - val_accuracy: 0.9992
Epoch 2/1000
118/118 [==============================] - 1s 5ms/step - loss: 2314.7263 - accuracy: 0.9110 - val_loss: 1948.6449 - val_accuracy: 0.9992
Epoch 3/1000
118/118 [==============================] - 0s 3ms/step - loss: 2239.7808 - accuracy: 0.9595 - val_loss: 1953.9216 - val_accuracy: 0.9992
Epoch 4/1000
118/118 [==============================] - 0s 4ms/step - loss: 2216.1807 - accuracy: 0.9817 - val_loss: 1957.0371 - val_accuracy: 0.9992
Epoch 5/1000
118/118 [==============================] - 1s 5ms/step - loss: 2180.4792 - accuracy: 0.9889 - val_loss: 1967.5446 - val_accuracy: 0.9992
Test Accuracy of Neural Network after one run: <keras.src.callbacks.History object at 0x7a1d23c739a0>
