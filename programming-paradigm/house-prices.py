# In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.
# So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, 
# a 2 bedroom house costs 150k etc. How would you create a neural network that learns this relationship so that it would 
# predict a 7 bedroom house as costing close to 400k etc.

# Import libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Create a one layer neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Measure loss and pass to optimizer to obtain next guess
model.compile(optimizer='sgd', loss='mean_squared_error')

# Use numpy to create two arrays (xs and ys)
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
 
# Pass the entire dataset in both forward and backward directions
# through the neural network 500 times
model.fit(xs, ys, epochs=500)

# Display the prediction that the neural network has
# for a 7 bedroom house
print(model.predict([7.0]))