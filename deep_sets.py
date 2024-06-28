from keras import layers, models, optimizers

import numpy as np
from keras import layers, models, optimizers
import numpy as np
import requests
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request


def permute_vector(vector):
    return np.random.permutation(vector)

def data_wrangle_S():
    Sweights, SHodge = [], []
    try:
        with open('Topological_Data.txt', 'r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx % 6 == 0:
                    Sweights.append(eval(line))
                if idx % 6 == 2:
                    SHodge.append(eval(line))
    except FileNotFoundError as e:
        import urllib.request
        urllib.request.urlretrieve('https://raw.githubusercontent.com/TomasSilva/MLcCY7/main/Data/Topological_Data.txt', 'Topological_Data.txt')
        with open('Topological_Data.txt', 'r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx % 6 == 0:
                    Sweights.append(eval(line))
                if idx % 6 == 2:
                    SHodge.append(eval(line))
    Sweights, SHodge = np.array(Sweights), np.array(SHodge)[:, 1:2]
    return Sweights, SHodge




#define new architecture for the NN
def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):

    inp = layers.Reshape((5, number_of_channels_in))(inp)
    # ---(1)---
    out1 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=False, activation='relu')(inp)
    # ---(2)---
    out2 = layers.AveragePooling1D(pool_size=5, strides=1, padding='valid')(inp)
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
    if pooling=='sum':
        p1 = layers.AveragePooling1D(5, strides=1, padding='valid')(e1)
    else:
        p1 = layers.MaxPooling1D(5, strides=1, padding='valid')(e1)
    p2 = layers.Flatten()(p1)

    # Repeat process for more layers
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    e2 = layers.Dropout(0.5)(e2)

    if pooling=='sum':
        p3 = layers.AveragePooling1D(5, strides=1, padding='valid')(e2)
    else:
        p3 = layers.MaxPooling1D(5, strides=1, padding='valid')(e2)
    p4 = layers.Flatten()(p3)

    fc1 = layers.Dense(64, activation='relu')(p4)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    out = layers.Dense(1, activation='linear')(fc2)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


def train_network(X_train, y_train, X_test, y_test):
    model = get_deep_sets_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    history = model.fit(
        X_train, y_train,
        epochs=999999,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return model, history

def daattavya_accuracy(weights, hodge_numbers, model):
    bound = 0.05*(np.max(hodge_numbers)-np.min(hodge_numbers)) #define the bound as done in Daattavya's paper
    random_indices = np.random.choice(np.array(weights).shape[0], 1000, replace=False) #make a selection as to not work with all the data
    random_selection = weights[random_indices]
    predictions = model.predict(random_selection)
    return np.mean(np.where(np.absolute(np.array(predictions)-hodge_numbers[random_indices]) < bound,1,0)) #use definition of accuracy as in paper


#running the program:

# test the data with neural network trained on ordered inputs
if __name__ == '__main__':
    #training on the sasakain hodge numbers
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #split data into training and testing
    X_test_permuted = permute_vector(X_test)
    # Train network on permuted data
    model, history = train_network(X_train, y_train, X_test, y_test)

    # Evaluate accuracy on original test set and permuted test set
    print('Accuracy as defined in the paper:')
    print(str(round(daattavya_accuracy(X_test, y_test, model) * 100, 1)) + '%')
    print(str(round(daattavya_accuracy(X_test_permuted, y_test, model) * 100, 1)) + '%')

    # Define the input vectors
    test_vector1 = np.array([1, 2, 3, 4, 5])
    test_vector2 = np.array([5, 4, 3, 2, 1])

    # Reshape the input vectors to match the input shape expected by the model
    test_vector1_reshaped = test_vector1.reshape(1, -1)
    test_vector2_reshaped = test_vector2.reshape(1, -1)

    # Get the predictions for the input vectors
    prediction1 = model.predict(test_vector1_reshaped)
    prediction2 = model.predict(test_vector2_reshaped)

    # Print the predictions
    print(f'Prediction for the vector [1, 2, 3, 4, 5]: {prediction1[0][0]}')
    print(f'Prediction for the vector [5, 4, 3, 2, 1]: {prediction2[0][0]}')
