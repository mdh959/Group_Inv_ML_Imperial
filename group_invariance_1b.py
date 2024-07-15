import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import itertools
from PrepworkSasakian import daattavya_accuracy, data_wrangle_S, permute_vector

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

def train_network(X_train, y_train, X_test, y_test, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    history = model.fit(
        X_train, y_train,
        epochs=999999,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return model, history

if __name__ == '__main__':
    # Training on the Sasakian Hodge numbers
    X, y = data_wrangle_S()
    X = X.reshape(-1, 5)  # Reshape the data to fit the model input shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing

    original_accuracies = []
    permuted_accuracies = []

    num_runs = 10

    for _ in range(num_runs):
        X_test_permuted = np.apply_along_axis(permute_vector, 1, X_test)
        model, history = train_network(X_train, y_train, X_test, y_test, build_model())

        original_accuracy = daattavya_accuracy(y_train, X_test, y_test, model)
        permuted_accuracy = daattavya_accuracy(y_train, X_test_permuted, y_test, model)

        original_accuracies.append(original_accuracy)
        permuted_accuracies.append(permuted_accuracy)

        print(f'Run {_ + 1}:')
        print(f'Accuracy on original test set: {original_accuracy * 100:.1f}%')
        print(f'Accuracy on permuted test set: {permuted_accuracy * 100:.1f}%')

    average_original_accuracy = np.mean(original_accuracies)
    average_permuted_accuracy = np.mean(permuted_accuracies)
    print(original_accuracies)

    print(f'\nAverage accuracy on original test set over {num_runs} runs: {average_original_accuracy * 100:.1f}%')
    print(f'Average accuracy on permuted test set over {num_runs} runs: {average_permuted_accuracy * 100:.1f}%')
