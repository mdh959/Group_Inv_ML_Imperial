import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Set the floating point precision
tf.keras.backend.set_floatx('float64')

class SineActivation(Layer):
    def __init__(self):
        super(SineActivation, self).__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2*np.pi*inputs), tf.cos(2*np.pi*inputs)], 1)

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

def partial_derivative(u, x, dim):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        u_val = NN(x)  # Directly use the input tensor `u`

    du_dx = tape.gradient(u_val, x)
    partial_derivative_dim = du_dx[:, dim]  # Extracting the partial derivative w.r.t specified dimension
    return partial_derivative_dim

def hodge_star_1_form(u):
    u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert u to a tf.Tensor
    u1, u2, u3 = u[:, 0:1], u[:, 1:2], u[:, 2:3]

    # Hodge star operation
    star_u = tf.concat([
        u1,   # Coefficient of dx1
        -u2,   # Coefficient of dx2
        u3    # Coefficient of dx3
    ], axis=1)

    return star_u

def hodge_star_2_form(d_u):
    d_u = tf.convert_to_tensor(d_u, dtype=tf.float64)  # Convert d_u to a tf.Tensor
    
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

def exterior_derivative_1_form(u, x):
    u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert u to a tf.Tensor
    du1_dx2 = partial_derivative(u[:, 0], x, 1)  # Partial derivative with respect to x2
    du1_dx3 = partial_derivative(u[:, 0], x, 2)  # Partial derivative with respect to x3
    du2_dx1 = partial_derivative(u[:, 1], x, 0)  # Partial derivative with respect to x1
    du2_dx3 = partial_derivative(u[:, 1], x, 2)  # Partial derivative with respect to x3
    du3_dx1 = partial_derivative(u[:, 2], x, 0)  # Partial derivative with respect to x1
    du3_dx2 = partial_derivative(u[:, 2], x, 1)  # Partial derivative with respect to x2

    d_u = tf.stack([du1_dx2, du1_dx3, -du2_dx1, du2_dx3, du3_dx1, -du3_dx2], axis=1)

    return d_u

def star_derivative_2_form(u, x):
    u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert u to a tf.Tensor
    du1_dx1 = partial_derivative(u[:, 0], x, 0)
    du2_dx2 = partial_derivative(u[:, 1], x, 1)
    du3_dx3 = partial_derivative(u[:, 2], x, 2)
    f = (du1_dx1 + du2_dx2 + du3_dx3)
    return f

def derivative_function(u, x):
    u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert f to a tf.Tensor
    df_dx1 = partial_derivative(u, x, 0)  # Partial derivative with respect to x1
    df_dx2 = partial_derivative(u, x, 1)  # Partial derivative with respect to x2
    df_dx3 = partial_derivative(u, x, 2)  # Partial derivative with respect to x3
    der_du = tf.stack([
        df_dx1,df_dx2,df_dx3
    ], axis=1)
    return der_du

def loss(x_collocation):
    x1 = (x_collocation[:, 0:1])
    x2 = (x_collocation[:, 1:2])
    x3 = (x_collocation[:, 2:3])
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x1)
        tape.watch(x2)
        tape.watch(x3)

        # Evaluate the model at the collocation points
        u = NN(x_collocation)

        # Calculate d  hodge_star  d  hodge_star
        hodge_star_du = hodge_star_1_form(u)
        hodge_star_d_hodge_star_du = star_derivative_2_form(hodge_star_du, x_collocation)
        d_hodge_star_d_hodge_star_du = derivative_function(hodge_star_d_hodge_star_du, x_collocation)

        # Calculate hodge_star  d  hodge_star  d
        d_u = exterior_derivative_1_form(u, x_collocation)
        hodge_star_d_u = hodge_star_2_form(d_u)
        d_hodge_star_d_u = exterior_derivative_1_form(hodge_star_d_u, x_collocation)
        hodge_star_d_hodge_star_d_u = hodge_star_2_form(d_hodge_star_d_u)
        
        # sum Right and Left of pde
        sum_tensor = hodge_star_d_hodge_star_d_u + d_hodge_star_d_hodge_star_du 
        
        # Compute the loss based on sum_tensor
        loss_value = tf.reduce_mean(tf.square(sum_tensor))

    return loss_value, sum_tensor

def train(x_collocation, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_history = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss_value, sum_tensor = loss(x_collocation)
        grads = tape.gradient(loss_value, NN.trainable_variables)
        optimizer.apply_gradients(zip(grads, NN.trainable_variables))
        loss_history.append(loss_value.numpy())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")
    
    # Print the sum_tensor after training
    print("Final sum_tensor:", sum_tensor.numpy())
    
    return loss_history

# Generate collocation points within a unit cube [0, 1] x [0, 1] x [0, 1]
x_collocation = np.array([[3.0, 3.0, 3.0],[1.0,1.0,1.0]], dtype=np.float64)
x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

# Training parameters
epochs = 1000
learning_rate = 0.001

# Train the model
loss_history = train(x_collocation, epochs, learning_rate)
