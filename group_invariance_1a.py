import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from PrepworkSasakian import get_network, data_wrangle_S, train_network, permute_vector

def create_group_invariant_function(model):
    """Create a group-invariant function from the given neural network model."""
    def psi(X):
        permutations = np.array(list(itertools.permutations(X)))
        num_permutations = min(len(permutations), 120)  # Limit the number of permutations
        permutations = permutations[:num_permutations]
        predictions = np.array([model.predict(np.expand_dims(p, axis=0))[0] for p in permutations])
        return np.mean(predictions, axis=0)
    return psi

def group_invariant_accuracy(training_outputs, test_inputs, test_outputs, psi, num_permutations=120):
    bound = 0.05 * (np.max(training_outputs) - np.min(training_outputs))  # Define the bound as in Daattavya's paper
    num_samples = test_inputs.shape[0]
    predictions = np.zeros(num_samples)

    for i in range(num_samples):
        permuted_inputs = np.array(list(itertools.permutations(test_inputs[i])))[:num_permutations]
        prediction = np.mean([psi(p) for p in permuted_inputs], axis=0)
        predictions[i] = prediction

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
