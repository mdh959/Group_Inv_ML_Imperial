import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import itertools
from PrepworkSasakian import daattavya_accuracy, data_wrangle_S, permute_vector, train_network

def build_model():
    input_layer = tf.keras.layers.Input(shape=(5,))

    # Generate all permutations of the input vector
    def permute_inputs(x):
        perms = list(itertools.permutations(range(5)))
        permuted_inputs = [tf.gather(x, indices=perm, axis=1) for perm in perms]
        return tf.stack(permuted_inputs, axis=1)

    permuted_inputs = tf.keras.layers.Lambda(permute_inputs)(input_layer)

    # Define shared model for parallel processing
    def create_shared_model():
        input_perm = tf.keras.layers.Input(shape=(5,))
        h1 = tf.keras.layers.Dense(16, activation='relu')(input_perm)
        h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
        h3 = tf.keras.layers.Dense(16, activation='relu')(h2)
        out = tf.keras.layers.Dense(1, activation='linear')(h3)
        return tf.keras.models.Model(input_perm, out)

    shared_nn = create_shared_model()

    # Apply the shared model to each permuted input
    permuted_outputs = tf.keras.layers.TimeDistributed(shared_nn)(permuted_inputs)


    # Sum the outputs of these individual models running in parallel
    summed_output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(permuted_outputs)

    # Define the overall model
    final_model = tf.keras.models.Model(inputs=input_layer, outputs=summed_output)

    final_model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )

    return final_model

if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing
    accuracy = []
    num_runs = 10
    for _ in range(num_runs):
      # Train network on permuted data
      model, history = train_network(X_train, y_train, X_test, y_test)
      accuracy.append(daattavya_accuracy(y_train, X_test, y_test, model))
    print(accuracy)
