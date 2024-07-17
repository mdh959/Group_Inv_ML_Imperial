import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request
import itertools

def train_network(X_train, y_train, X_test, y_test, model):
    print(model.summary()) # print an overview of the neural network created
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return model, history

def data_wrangle_CNI():
    weights, CNI = [], []
    try:
        with open('Data/Topological_Data.txt', 'r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx % 6 == 0: weights.append(eval(line))
                if idx % 6 == 3: CNI.append(eval(line))
    except FileNotFoundError as e:
        urllib.request.urlretrieve('https://raw.githubusercontent.com/TomasSilva/MLcCY7/main/Data/Topological_Data.txt', 'Topological_Data.txt')
        with open('Topological_Data.txt', 'r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx % 6 == 0: weights.append(eval(line))
                if idx % 6 == 3: CNI.append(eval(line))

    weights, CNI = np.array(weights), np.array(CNI)[:, np.newaxis]
    return weights, CNI

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
        out = tf.keras.layers.Dense(24, activation='softmax')(h3)
        return tf.keras.models.Model(input_perm, out)

    shared_nn = create_shared_model()

    # Apply the shared model to each permuted input
    permuted_outputs = tf.keras.layers.TimeDistributed(shared_nn)(permuted_inputs)

    # Sum the outputs of these individual models running in parallel
    summed_output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(permuted_outputs)

    # Define the overall model
    final_model = tf.keras.models.Model(inputs=input_layer, outputs=summed_output)

    final_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    return final_model

def classification_accuracy(test_inputs, test_outputs, model):
    predictions = model.predict(test_inputs)
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == test_outputs)

if __name__ == '__main__':
    X, y = data_wrangle_CNI()
    y = ((y - 1) / 2).astype(int).flatten()
    accuracy = []
    num_runs = 10
    for _ in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjusted to 80-20 split
        model, history = train_network(X_train, y_train, X_test, y_test, build_model())  # Train network on chosen data
        accuracy.append(classification_accuracy( X_test, y_test, model))
    print(accuracy)
