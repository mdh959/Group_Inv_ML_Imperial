import requests
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from prepwork import data_wrangle_CNI, get_classifier
from PrepworkSasakian import train_network
import itertools

################################################################################
# defining a new accuracy which is group invariant: 

def classification_accuracy(test_inputs, test_outputs, model):
    num_samples = test_inputs.shape[0]
    predictions = np.zeros(test_inputs.shape[0])
    
    for i in range(num_samples):
        # Generate all permutations of the current test sample
        permuted_inputs = np.array(list(itertools.permutations(test_inputs[i]))).reshape(-1, 5)
        pred = model.predict(permuted_inputs)
        predictions[i] = np.mean(np.argmax(pred, axis=1))
    
    return np.mean(predictions == test_outputs)
 

if __name__ == '__main__':
    X, y = data_wrangle_CNI()
    y = ((y - 1) / 2).astype(int).flatten()
    accuracy = []
    num_runs = 1
    for _ in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjusted to 80-20 split
        model, history = train_network(X_train, y_train, X_test, y_test, get_classifier())  # Train network on chosen data
        accuracy.append(classification_accuracy( X_test, y_test, model))
    print(accuracy)
