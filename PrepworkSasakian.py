import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

################################################################################
#importing and wrangling data

def data_wrangle_S():
    Sweights, SHodge = [], []
    try:
        with open('Topological_Data.txt','r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx%6 == 0: Sweights.append(eval(line))
                if idx%6 == 2: SHodge.append(eval(line))
    except FileNotFoundError as e:
        urllib.request.urlretrieve('https://raw.githubusercontent.com/TomasSilva/MLcCY7/main/Data/Topological_Data.txt', 'Topological_Data.txt')
        with open('Topological_Data.txt','r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx%6 == 0: Sweights.append(eval(line))
                if idx%6 == 2: SHodge.append(eval(line))
    Sweights, SHodge = np.array(Sweights), np.array(SHodge)[:, 1:2]
    return Sweights, SHodge

################################################################################
#defining and training NN

def get_network():
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Flatten()(inp)
    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(16, activation='relu')(h2)
    out = tf.keras.layers.Dense(1, activation='linear')(h3)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics = ['accuracy']
    )
    return model

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

################################################################################
#defining accuracy as in the paper

def daattavya_accuracy(weights, hodge_numbers, model):
    bound = 0.05*(np.max(hodge_numbers)-np.min(hodge_numbers)) #define the bound as done in Daattavya's paper
    random_indices = np.random.choice(np.array(weights).shape[0], 1000, replace=False) #make a selection as to not work with all the data
    random_selection = weights[random_indices] 
    predictions = model.predict(random_selection)
    return np.mean(np.where(np.absolute(np.array(predictions)-hodge_numbers[random_indices]) < bound,1,0)) #use definition of accuracy as in paper

################################################################################
#running the program: 

if __name__ == '__main__':
    #training on the sasakain hodge numbers, as in the paper
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) #split data into training and testing
    # Permute the training data
    permuted_X_train = np.apply_along_axis(permute_vector, 1, X_train)
    X_test_permuted = permute_vector(X_test)
    # Train network on permuted data
    model, history = train_network(permuted_X_train, y_train, X_test, y_test)
    
    # Evaluate accuracy on original test set
    print('Accuracy as defined in the paper:')
    print(str(round(daattavya_accuracy(X_test, y_test, model) * 100, 1)) + '%')
    print(str(round(daattavya_accuracy(X_test_permuted, y_test, model) * 100, 1)) + '%')

if __name__ == '__main__':
    #training on the sasakain hodge numbers, as in the paper
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) #split data into training and testing
    # Permute the training data
    permuted_X_train = np.apply_along_axis(permute_vector, 1, X_train)
    X_test_permuted = permute_vector(X_test)
    # Train network on permuted data
    model, history = train_network(X_train, y_train, X_test, y_test)
    
    # Evaluate accuracy on original test set
    print('Accuracy as defined in the paper:')
    print(str(round(daattavya_accuracy(X_test, y_test, model) * 100, 1)) + '%')
    print(str(round(daattavya_accuracy(X_test_permuted, y_test, model) * 100, 1)) + '%')
