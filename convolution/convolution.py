# For every pixel, take its value, and take a look at the value of 
# its neighbors. If our filter is three by three, then we can take 
# a look at the immediate neighbor, so that you have a corresponding
# three by three grid. Then to get the new value for the pixel, we 
# simply multiply each neighbor by the corresponding value in the filter.

# Some convolutions will change the image in such a way that certain 
# features in the image get emphasized.

# We don't have to do all the math for filtering and compressing,
# we simply define convolutional and pooling layers to do the job for us.
model = tf.keras.models.Sequential([
    # Ask Keras to generate 64 3x3 filters with a relu 
    # activation to throw the negative values away
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',
    # Seek an input shape of 28x28 using a single byte for
    # color depth
        input_shape=(28,28,1)),
    # Create a maximum 2x2 pooling layer; so for every
    # four pixels, the biggest one will survive
    tf.keras.layers.MaxPooling2D(2,2),  
    # Add another convolutional layer,
    # and another max-pooling layer so that the network can
    # learn another set of convolutions on top of the existing 
    # one, and then again, pool to reduce the size.
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the output shape from (None,5,5,64) to (None,1600)
    tf.keras.layers.Flatten(),
    # Just the regular densely-connected NN layers
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='software')
])