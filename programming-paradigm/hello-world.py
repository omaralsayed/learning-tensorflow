# Import tensorflow and numpy libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

# A neural network is basically a set of functions which can learn patters.
# In keras, you use the word dense to define a layer of connected neurons. 
# There's only one dense here. So there's only one layer and there's only one 
# unit in it, so it's a single neuron. Successive layers are defined in sequence, 
# hence the word sequential.
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# The loss function measures the loss and then gives the data to the optimizer which
# figures out the next guess.
# The logic is that each guess should be better than the one before. As the 
# guesses get better and better, an accuracy approaches 100 percent, the 
# term "convergence" is used.
# The loss is "mean squared error" and the optimizer is "SGD" which stands for 
# Stochastic Gradient Descent.
model.compile(optimizer='sgd', loss='mean_squared_error')

# The np.array is using a Python library called numpy that makes data 
# representation particularly enlists much easier.
# We are asking the model to figure out how to fit the X values to the Y values.
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

# Epochs of value 500 means that it will go through the training loop 500 times.
model.fit(xs, ys, epochs=500)

# We can ask the neural network to make a guess using the predict method.
print(model.predict([10.0]))

# The 'Hello World' of neural networks summary:
# Using mental math we can understand the pattern and map x onto y with the linear
# equation Y = 2X - 1.
# The training data is very small, so the output is 18.984648 instead of 19.0.
# Even though there is a very high probability that Y = 19.0 for X = 10.0, the neural
# network is not 100% positive.