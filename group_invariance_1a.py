import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request
import itertools
from PrepworkSasakian import get_network, data_wrangle_S, train_network


if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_S()
    X = X.reshape(-1, 5)  # Reshape the data to fit the model input shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing
    model = get_network()

    # Train network on permuted data
    model, history = train_network(X_train, y_train, X_test, y_test, model)

    # Evaluate accuracy on original test set and permuted test set
    print('Accuracy as defined in the paper:')
    print(str(round(GI_1a(y_train, X_test, y_test, model) * 100, 1)) + '%')
