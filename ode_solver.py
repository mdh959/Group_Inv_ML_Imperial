import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.layers import Layer

tf.keras.backend.set_floatx('float64')

def sine_activation(x):
    return tf.sin(2*math.pi*x)

class SineActivation(Layer):
    def __init__(self):
        super(SineActivation, self).__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2*math.pi*inputs), tf.cos(2*math.pi*inputs)], 1)

NN = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        SineActivation(),
        tf.keras.layers.Dense(units=64, activation='sigmoid'),
        tf.keras.layers.Dense(units=1, activation=None),
    ])

NN.summary()

optm = tf.keras.optimizers.Adam(learning_rate=0.001)

def ode_system(x, net):
    # Define spatial points for evaluation
    x = tf.reshape(x, [-1, 1])
    x_0 = tf.constant([[0]], dtype=tf.float64)
    x_2pi = tf.constant([[2 * np.pi]], dtype=tf.float64)
    #x = tf.linspace(0.0, 2.0 * np.pi, num=1000)


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
    IC_loss = tf.square(net(x_0) - 0) + tf.square(net(x_2pi) - 0)

    square_loss = tf.square(ode_loss) 
    total_loss = tf.reduce_mean(square_loss)

    del tape1, tape2

    return total_loss, IC_loss

def x_function(x):
    return (1/280) * tf.sin(2 * np.pi * x)
test_x = np.linspace(0, 1, 100).astype(np.float64)
total, ode = ode_system(test_x, x_function)

def NN_derivative(x, net):
    """
    Compute the second derivative of the neural network `net` with respect to `x`.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    x = tf.reshape(x, [-1, 1])
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with tf.GradientTape() as tape_inner:
            tape_inner.watch(x)
            u = net(x)
        u_x = tape_inner.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    
    del tape_inner
    return u_xx

def F_derivative(x, net):
    """
    Compute the value of the derivative function:
    - f''(x) - sin(x)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    x = tf.reshape(x, [-1, 1])
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with tf.GradientTape() as tape_inner:
            tape_inner.watch(x)
            f = net(x)
        f_x = tape_inner.gradient(f, x)
    f_xx = tape.gradient(f_x, x)
    
    del tape_inner
    result = f_xx - tf.sin(x)
    
    return result


def Lipschitz_constant(x, net):
    """
    Calculate the Lipschitz constant for the PINN.
    """
    constant = NN_derivative(x, net)* F_derivative(x, net)
    return constant

def calculate_residuals(x, net):
    _, residuals = ode_system(x, net)
    return residuals

def lower_bound_int(net):
    """
    Estimate the lower bound for the integral based on neural network predictions.
    """
    x_values = np.linspace(0, 2*np.pi, 1000).astype(np.float64)
    f_values = net.predict(x_values).ravel()
    k = NN_derivative(x_values, net)

    lower_sum = -np.sum(k) / 1000000 + np.sum(f_values) / 1000
    
    return lower_sum


def upper_bound_int(net):
    """
    Estimate the upper bound for the integral based on neural network predictions.
    """
    x_values = np.linspace(0, 2*np.pi, 1000).astype(np.float64)
    k = NN_derivative(x_values, net)
    f_values = net.predict(x_values).ravel()
    
    upper_sum = np.sum(k) / 1000000 + np.sum(f_values) / 1000
    
    return upper_sum



# Start training
train_x = np.linspace(0, 2*np.pi, 10001).reshape(-1, 1).astype(np.float64)
train_loss_record = []

optm = tf.keras.optimizers.Adam(learning_rate=0.001)

for itr in range(1001):
    with tf.GradientTape() as tape:
        train_loss, _ = ode_system(tf.constant(train_x), NN)
        train_loss_record.append(train_loss.numpy())

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 100 == 0:
        print(f"Iteration {itr}, Loss: {train_loss.numpy()}")

# Save the model
tf.keras.models.save_model(NN, 'best_model.keras')

# Plot training loss
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Iterations')
plt.savefig("training_loss_plot.png")  # Save plot

# Error Analysis
x_i = np.linspace(0, 2*np.pi, 1001).astype(np.float64)

lipschitz_constant = Lipschitz_constant(x_i, NN) # Calculate the Lipschitz constant

error_bounds = []   #Find the error bounds
for i,x in enumerate(x_i):
    error_bound = lipschitz_constant[i][0] * 1/1000
    error_bounds.append(error_bound)

error_bounds = np.array(error_bounds)

residuals = calculate_residuals(x_i, NN).numpy().ravel()

# |F(f_approx)(x)| < | F(f)(x_i)| (Known) + | F(f)(x) - F(f)(x_i) |
total_errors = np.abs(residuals) + error_bounds[:len(residuals)]

errors_in_l2 = np.sum(total_errors ** 2) * 1/1000
print(f"Error in L2: {errors_in_l2}")

#Plot the error bounds
plt.figure(figsize=(10, 8))
plt.plot(x_i[:len(residuals)], residuals, '--r', label=r'bound for |F($f_{approx} (x_k)$)|')
plt.plot(x_i[:len(residuals)], error_bounds[:len(residuals)], ':b', label=r'bound for |$F(f_{approx})(x_k) - F(f_{approx})(x)$|')
plt.plot(x_i[:len(residuals)], total_errors, '-g', label=r'Total Error: $\delta_k$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Error Analysis(n = 1000)')
plt.savefig("Error.png")

# Generate and plot predictions
test_x = np.linspace(0, 2*np.pi, 100).astype(np.float64)
pred_u = NN.predict(test_x).ravel()
int_bound =  lower_bound_int(NN)
print(int_bound)
plt.plot(test_x, pred_u, '-.r', label=r'$f_{approx}(x)$')
plt.plot(test_x,  pred_u - int_bound, '--g', label=r'$\tilde{f} (x)$')
#plt.plot(test_x, 0 * test_x, '--k', label='zero_solution')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximate solution Plot')
plt.savefig("PINN_result.png")  # Save plot
plt.show()

plt.figure(figsize=(10, 8))
#plt.plot(test_x, 0 * test_x , '-k', label='Zero Solution', linewidth=3, alpha=0.8)
plt.plot(test_x, pred_u, '--r', label=r'$\tilde{f}(x)$', linewidth=1, alpha=1)
plt.ylim(-1, 1)
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\tilde{f} (x)$')
plt.title('Approximate solution Plot')
plt.savefig("result_axis.png")
