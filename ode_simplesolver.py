import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

tf.keras.backend.set_floatx('float64')

NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(units = 64, activation = 'sigmoid'),
    #tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    #tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    tf.keras.layers.Dense(units = 1)
])

NN.summary()

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)


def ode_system(x, net):
    # Define spatial points for evaluation
    x = tf.reshape(x, [-1, 1])
    x_0 = tf.constant([[0]], dtype=tf.float64)
    x_2pi = tf.constant([[2 * np.pi]], dtype=tf.float64)


    # First Gradient Calculation (f_x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            f = net(x)
        f_x = tape2.gradient(f, x)
    f_xx = tape1.gradient(f_x, x)

    # ODE Loss: f''(x) - sin(x) should be zero
    ode_loss = f_xx - tf.sin(x)

    # Initial Condition Loss (assuming f(0) = 0 and f(2pi) = 0
    IC_loss = net(x_0) - 0 + net(x_2pi) - 0

    square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    total_loss = tf.reduce_mean(square_loss)
    del tape1, tape2

    return total_loss

train_x = np.linspace(0, 2*np.pi, 6000).reshape(-1, 1).astype(np.float64)
train_loss_record = []

for itr in range(6000):
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_x, NN)
        train_loss_record.append(train_loss)

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 == 0:
        print(train_loss.numpy())

plt.figure(figsize = (10,8))
plt.plot(train_loss_record)
plt.show()

test_x = np.linspace(0, 2*np.pi, 100).astype(np.float64)

train_f = -np.sin(2*np.pi*train_x)
true_f = -np.sin(2*np.pi*test_x)
pred_f = NN.predict(test_x).ravel()

plt.figure(figsize = (10,8))
plt.plot(train_x, train_f, 'ok', label = 'Train')
plt.plot(test_x, true_f, '-k',label = 'True')
plt.plot(test_x, pred_f, '--r', label = 'Prediction')
plt.legend(fontsize = 15)
plt.xlabel('x', fontsize = 15)
plt.ylabel('f', fontsize = 15)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

# Define the PINN class for solving f''(x) = sin(x)
class PINN:
    def __init__(self):
        # Define the neural network model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((1,)),
            tf.keras.layers.Dense(units=64, activation='sigmoid'),
            tf.keras.layers.Dense(units=1)
        ])

        # Define the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def f_model(self, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            # Getting the prediction
            f = self.model(x)

            # Derivatives
            f_x = tape.gradient(f, x)
        f_xx = tape.gradient(f_x, x)

        del tape

        return f, f_xx

    def ode_loss(self, x):
        x = tf.reshape(x, [-1, 1])
        x_0 = tf.constant([[0]], dtype=tf.float64)
        x_2pi = tf.constant([[2 * np.pi]], dtype=tf.float64)
        f, f_xx = self.f_model(x)

        # Right-hand side of the differential equation
        rhs = tf.sin(x)

        # ODE Loss: f''(x) - sin(x) should be zero
        ode_loss = f_xx - rhs

        # Initial Condition Loss (assuming f(0) = 0 and f(2pi) = 0
        IC_loss = tf.square(self.model(x_0) - 0) + tf.square(self.model(x_2pi) - 0)

        square_loss = tf.square(ode_loss) + tf.square(IC_loss)
        total_loss = tf.reduce_mean(square_loss)
        

        return total_loss

    def train(self, train_x, iterations=10000):
        train_loss_record = []

        for itr in range(iterations):
            with tf.GradientTape() as tape:
                loss = self.ode_loss(train_x)

            train_loss_record.append(loss.numpy())
            grad_w = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad_w, self.model.trainable_variables))

            if itr % 1000 == 0:
                print(f"Iteration {itr}: Loss = {loss.numpy()}")

        plt.figure(figsize=(10, 8))
        plt.plot(train_loss_record)
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Training Loss', fontsize=15)
        plt.show()

    def predict(self, test_x):
        test_x = tf.reshape(test_x, [-1, 1])

        # Predict using the trained model
        predictions = self.model(test_x).numpy().ravel()

        return predictions

# Main script to run the PINN
if __name__ == "__main__":
    # Create PINN instance
    pinns_solver = PINN()

    # Training data
    train_x = np.linspace(0, 2*np.pi, 100).reshape(-1, 1).astype(np.float64)

    # Train the PINN
    pinns_solver.train(train_x, iterations=10000)

    # Test data
    test_x = np.linspace(0, 2*np.pi, 100).reshape(-1, 1).astype(np.float64)

    # Predict and plot results
    predictions = pinns_solver.predict(test_x)

    plt.figure(figsize=(10, 8))
    plt.plot(test_x, -np.sin(test_x), '-k', label='True')
    plt.plot(test_x, predictions, '--r', label='Prediction')
    plt.legend(fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('f', fontsize=15)
    plt.show()
