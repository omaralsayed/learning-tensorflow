# Machine Learning Fairness:
# Developers.Google.com/machine-learning/fairness-overview/

# Use a lot of pictures and label them for the machine
# to learn what they represent and figure out the pattern.
# There is a lot of free datasets with millions of labelled
# pictures, such as Fashion MNIST.

# In Fashion MNIST, the pictures are scaled down to 28px by
# 28px. Usually, the smaller the better, because the computer
# has less processing to do.

# Machine Learning depends on having good data to train a system with.
# The approach to achieve Computer Vision is similar to that of
# creating a simple one nueron neural network, the only big difference
# is the dataset.

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Fashion MNIST dataset is available with an API call in TensorFlow.
fashion_mnist = keras.datasets.fashion_mnist
# If we call the load_data() method, it will return four lists to us.
# That is training data, training labels, testing data, and testing labels.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# It is a good practice to train the neural network on some data and then
# give the model some other data that it has not previously seen, to see
# how well or bad the neural network if performing (can be done by training
# the neural net with most of the data and testing it with the remaining
# data).

# Build three sequential layers
model = keras.Sequential([
    # Flatten layer with the input shaping 28 by 28
    # This layer is specifing the shape that we are
    # expecting our data to be in
    keras.layers.Flatten(input_shape=(28, 28)),
    # Hidden layer
    keras.layers.Dense(128, activation=tf.nn.relu),
    # 10 neurons because we have 10 classes of clothing
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Colab Workbook
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# First image in array
plt.imshow(train_images[0])
print(training_labels[0])
print(training_images[0])

# Neural networks work better with normalized data

# It is easier to treat all values as between 0 and 1 (normalized data)
# No loop necessary to normalize data in Python (entire array can be divided)
training_images = training_images / 255.0
test_images = test_images / 255.0

# Design the model
# Transforms the square (picture) onto a one dimensional set
model = tf.keras.models.Sequential([
    tf.keras.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])