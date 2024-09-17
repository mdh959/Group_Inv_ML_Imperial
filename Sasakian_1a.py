import numpy as np
import itertools
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

def permute_vector(vector):
    return np.random.permutation(vector)
def daattavya_accuracy(training_outputs, test_inputs, test_outputs, model):
    bound = 0.05 * (np.max(training_outputs) - np.min(training_outputs))
    predictions = model.predict(test_inputs)
    return np.mean(np.where(np.abs(np.array(predictions) - test_outputs) < bound, 1, 0))
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

def create_group_invariant_function(model):
    """Create a group-invariant function from the given neural network model."""
    def psi(X):
        # Generate all 120 permutations
        permutations = np.array(list(itertools.permutations(X)))
        # Predict for all permutations at once
        predictions = model.predict(permutations)
        # Return the mean of the predictions
        return np.mean(predictions, axis=0)
    return psi

def group_invariant_accuracy(training_outputs, test_inputs, test_outputs, psi):
    bound = 0.05 * (np.max(training_outputs) - np.min(training_outputs))  # Define the bound as in Daattavya's paper
    num_samples = test_inputs.shape[0]
    predictions = np.zeros(test_inputs.shape[0])
    for i in range(num_samples):
        permuted_inputs = np.array(list(itertools.permutations(test_inputs[i]))).reshape(120, -1)
        predictions[i] = np.mean(psi(permuted_inputs))
    return np.mean(np.where(np.abs(predictions - test_outputs.flatten()) < bound, 1, 0))


if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_S()
    X = X.reshape(-1, 5)  # Reshape the data to fit the model input shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing
    model = get_network()

    # Train network on permuted data
    model, history = train_network(X_train, y_train, X_test, y_test)
    X_test_permuted = permute_vector(X_test)

    # Create group-invariant function psi
    psi = create_group_invariant_function(model)

    # Evaluate accuracy using the new group-invariant function
    accuracy = group_invariant_accuracy(y_train, X_test, y_test, psi)
    print(f'Accuracy as defined in the paper with the new group-invariant function: {accuracy * 100:.1f}%')

    permuted_accuracy = group_invariant_accuracy(y_train, X_test_permuted, y_test, psi)
    print(f'Permuted accuracy: {permuted_accuracy * 100:.1f}%')
