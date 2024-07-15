import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Set the floating point precision
tf.keras.backend.set_floatx('float64')

class SineActivation(Layer):
    def __init__(self):
        super(SineActivation, self).__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(inputs), tf.cos(inputs)], 1)

# Define the neural network model
NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((3,)),  # input layer with 3 inputs: x1, x2, x3
    SineActivation(),
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=64, activation='tanh'),
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=3)  # Output layer for f1, f2, f3
])

NN.summary()

# Define basis vectors and one-forms in the 3-dimensional torus space
class TorusSpace:
    def __init__(self):
        self.base_vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        self.base_oneforms = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]


def partial_derivative(u, x, dim):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u_val = u  # Directly use the input tensor `u`

    du_dx = tape.gradient(u_val, x)
    partial_derivative_dim = du_dx[:, dim]  # Extracting the partial derivative w.r.t specified dimension
    return partial_derivative_dim

# Define Hodge star operation on 1-forms
def hodge_star_1_form(u):
    u1, u2, u3 = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    
    # Hodge star operation
    star_u = u1 - u2  + u3
    
    return star_u

def hodge_star_2_form(d_u):
    # Extract components from d_u
    df2_dx3 = d_u[:, 3:4]
    df3_dx2 = d_u[:, 5:6]
    df3_dx1 = d_u[:, 4:5]
    df1_dx3 = d_u[:, 1:2]
    df1_dx2 = d_u[:, 0:1]
    df2_dx1 = d_u[:, 2:3]

    # Compute f1_prime, f2_prime, f3_prime
    f1_prime = df2_dx3 - df3_dx2
    f2_prime = df3_dx1 - df1_dx3
    f3_prime = df1_dx2 - df2_dx1

    # Construct the 1-form (output)
    star_d_u = tf.concat([
        f1_prime,   # Coefficient of dx1
        f2_prime,   # Coefficient of dx2
        f3_prime    # Coefficient of dx3
    ], axis=1)

    return star_d_u


# Define the exterior derivative (d) for a 1-form in R^3
def exterior_derivative_1_form(u, x):
    du1_dx2 = partial_derivative(u[:, 0], x, 1)  # Partial derivative with respect to x2
    du1_dx3 = partial_derivative(u[:, 0], x, 2)  # Partial derivative with respect to x3
    du2_dx1 = partial_derivative(u[:, 1], x, 0)  # Partial derivative with respect to x1
    du2_dx3 = partial_derivative(u[:, 1], x, 2)  # Partial derivative with respect to x3
    du3_dx1 = partial_derivative(u[:, 2], x, 0)  # Partial derivative with respect to x1
    du3_dx2 = partial_derivative(u[:, 2], x, 1)  # Partial derivative with respect to x2

    d_u = du1_dx2 + du1_dx3 
    d_u += -(du2_dx1) + (du2_dx3 )
    d_u += (du3_dx1 ) - (du3_dx2)
    return d_u

def star_derivative_2_form(u, x):
    """ Computes hodge_star * d on a 2-form in R^3. """
    du1_dx1 = partial_derivative(u[:, 0], x, 1)
    du2_dx2 = partial_derivative(u[:, 1], x, 2)
    du3_dx3 = partial_derivative(u[:, 2], x, 3)
    f = (du1_dx1 + du2_dx2 + du3_dx3)
    return f

def derivative_function(f, x):
    df_dx1 = partial_derivative(f, x, 0)  # Partial derivative with respect to x1
    df_dx2 = partial_derivative(f, x, 1)  # Partial derivative with respect to x2
    df_dx3 = partial_derivative(f, x, 2)  # Partial derivative with respect to x3
    der_du = tf.concat([
        df_dx1,   # Coefficient of dx1
        df_dx2,   # Coefficient of dx2
        df_dx3    # Coefficient of dx3
    ], axis=1)

    return der_du 

class PINN:
    def __init__(self, model):
        self.model = model
        self.torus_space = TorusSpace
    def loss(self, x_collocation):
        x1 = x_collocation[:, 0:1]
        x2 = x_collocation[:, 1:2]
        x3 = x_collocation[:, 2:3]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x1)
            tape.watch(x2)
            tape.watch(x3)

            # Evaluate the model at the collocation points
            u = self.model(x_collocation)

            # Calculate d  hodge_star  d  hodge_star
            hodge_star_du = hodge_star_1_form(u)
            hodge_star_d_hodge_star_du = star_derivative_2_form(hodge_star_du, x_collocation)
            d_hodge_star_d_hodge_star_du = derivative_function(hodge_star_d_hodge_star_du, x_collocation)

            # Calculate hodge_star  d  hodge_star  d
            d_u = exterior_derivative_1_form(u, x_collocation)
            hodge_star_d_u = hodge_star_2_form(d_u)
            d_hodge_star_d_u = exterior_derivative_1_form(hodge_star_d_u, x_collocation)
            hodge_star_d_hodge_star_d_u = hodge_star_2_form(d_hodge_star_d_u, x_collocation)
            # sum Right and Left of pde
            sum_tensor = d_hodge_star_d_hodge_star_du + hodge_star_d_hodge_star_d_u
            # Compute the loss based on sum_tensor
            loss = tf.reduce_mean(tf.square(sum_tensor))

        return loss

    def train(self, x_collocation, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss_value = self.loss(x_collocation)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

# Generate collocation points within a unit cube [0, 1] x [0, 1] x [0, 1]
num_samples_collocation = 1000
x_collocation = np.random.uniform(low=0, high=1, size=(num_samples_collocation, 3))

# Convert to tensors
x_collocation_tf = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

# Initialize PINN model
pinn = PINN(NN)

# Training parameters
epochs = 1000
learning_rate = 0.001

# Train the model
pinn.train(x_collocation_tf, epochs, learning_rate)
