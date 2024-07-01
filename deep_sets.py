import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

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

def permute_vector(vector):
    return np.random.permutation(vector)

def daattavya_accuracy(training_outputs, test_inputs, test_outputs, model):
    bound = 0.05 * (np.max(training_outputs) - np.min(training_outputs))
    predictions = model.predict(test_inputs)
    return np.mean(np.where(np.abs(np.array(predictions) - test_outputs) < bound, 1, 0))


# Define equivariant layer
def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):
    inp = layers.Reshape((5, number_of_channels_in))(inp)
    out1 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=True, activation='relu')(inp)
    return out1

# Define Deep Sets network with permutation invariant design
def get_deep_sets_network():
    number_of_channels = 100
    inp = layers.Input(shape=(5,))
    
    # Apply equivariant layer
    e1 = equivariant_layer(inp, 1, number_of_channels)
    e1 = layers.Dropout(0.5)(e1)
    
    p1 = layers.AveragePooling1D(pool_size=5, strides=1, padding='valid')(e1)
    p2 = layers.Flatten()(p1)

    # Sum the mapped elements (permutation invariant operation)
    #summed = layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]))(e1)
    
    #fc1 = layers.Dense(64, activation='relu')(summed)
    
    fc1 = layers.Dense(64, activation='relu')(p2)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    out = layers.Dense(1, activation='linear')(fc2)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

# Function to train the network
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

if __name__ == '__main__':
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
