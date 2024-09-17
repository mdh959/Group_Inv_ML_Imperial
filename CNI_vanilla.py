import requests
from PrepworkSasakian import train_network
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

def data_wrangle_CNI():
    weights, CNI = [], []
    try:
        with open('Data/Topological_Data.txt','r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx%6 == 0: weights.append(eval(line))
                if idx%6 == 3: CNI.append(eval(line))
    except FileNotFoundError as e:
        urllib.request.urlretrieve('https://raw.githubusercontent.com/TomasSilva/MLcCY7/main/Data/Topological_Data.txt', 'Topological_Data.txt')
        with open('Topological_Data.txt','r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx%6 == 0: weights.append(eval(line))
                if idx%6 == 3: CNI.append(eval(line))

    weights, CNI = np.array(weights), np.array(CNI)[:, np.newaxis]
    return weights, CNI

def get_classifier():
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Flatten()(inp)
    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(16, activation='relu')(h2)
    out = tf.keras.layers.Dense(24, activation='softmax')(h3)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics = ['accuracy']
    )
    return model   
# define accuracy

def classification_accuracy(test_inputs, test_outputs, model):
    predictions = model.predict(test_inputs)
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(np.where(np.array(predicted_classes) == test_outputs,1,0)) 
# run program

if __name__ == '__main__':
    X,y = data_wrangle_CNI()
    y = (y-1)/2
    y = np.round(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data into training and testing
    model, history = train_network(X_train, y_train, X_test, y_test, get_classifier()) # train network on chosen data
    print('Accuracy: ' + str(round(classification_accuracy(X_test, y_test, model)*100, 1)) + '%')
