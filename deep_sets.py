import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

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

def permute_vector(vector):
    # Shuffle the vector using NumPy's permutation function
    permuted_vector = np.random.permutation(vector)

    return permuted_vector
    
def daattavya_accuracy(weights, hodge_numbers, model):
    bound = 0.05*(np.max(hodge_numbers)-np.min(hodge_numbers)) #define the bound as done in Daattavya's paper
    random_indices = np.random.choice(np.array(weights).shape[0], 1000, replace=False) #make a selection as to not work with all the data
    random_selection = weights[random_indices] 
    predictions = model.predict(random_selection)
    return np.mean(np.where(np.absolute(np.array(predictions)-hodge_numbers[random_indices]) < bound,1,0)) #use definition of accuracy as in paper


def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):
    # Reshape input to (1, 5, number_of_channels_in)
    inp = layers.Reshape((1, 5, number_of_channels_in))(inp)

    # Convolutional layer with (1, 1) filter
    out1 = layers.Conv2D(number_of_channels_out, (1, 1), padding='valid', use_bias=False, activation='relu')(inp)

    # Average pooling over (1, 5) window
    out4 = layers.AveragePooling2D((1, 5), strides=(1, 1), padding='valid')(inp)

    # Concatenate along channels axis
    out4 = layers.Concatenate(axis=3)([out4] * 5)

    # Convolutional layer with (1, 1) filter after concatenation
    out4 = layers.Conv2D(number_of_channels_out, (1, 1), strides=(1, 1), padding='valid', use_bias=True, activation='relu')(out4)

    return layers.Add()([out1, out4])

def get_network(pooling='sum'):
    number_of_channels = 100
    inp = layers.Input(shape=(1, 5))
    inp_list = [inp for _ in range(number_of_channels)]
    inp_duplicated = layers.Concatenate(axis=2)(inp_list)
    # First equivariant layer
    e1 = equivariant_layer(inp_duplicated, number_of_channels, number_of_channels)
    
    # Second equivariant layer
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    
    if pooling == 'sum':
        p1 = layers.AveragePooling2D((1, 5), padding='valid')(e2)
    else:
        p1 = layers.MaxPooling2D((1, 5), padding='valid')(e2)
    
    # Flatten for fully connected layers
    p2 = layers.Flatten()(p1)
    
    # Fully connected layers
    fc1 = layers.Dense(64, activation='relu')(p2)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    
    # Output layer
    out = layers.Dense(1, activation='linear')(fc2)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

def train_network(X_train, y_train, X_test, y_test):
    model = get_S_network()
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_test, y_test),
        batch_size=1
    )
    return model, history

# test the data with neural network trained on ordered inputs
if __name__ == '__main__':
    #training on the sasakain hodge numbers, as in the paper
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) #split data into training and testing
    X_test_permuted = permute_vector(X_test)
    # Train network on permuted data
    model, history = train_network(X_train, y_train, X_test, y_test)
    
    # Evaluate accuracy on original test set
    print('Accuracy as defined in the paper:')
    print(str(round(daattavya_accuracy(X_test, y_test, model) * 100, 1)) + '%')
    print(str(round(daattavya_accuracy(X_test_permuted, y_test, model) * 100, 1)) + '%')
