import tensorflow as tf

# Every epoch (hyperparameter), you can callback to a code function, 
# having checked the metrics. If they're what you want to say, then 
# you can cancel the training at that point. 

# Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.4):
            print("\nCancelling: reached 60% accuracy")
            self.model.stop_training = True

# Analysis:
# It's implemented as a separate class, but that can be in-line 
# with the other code. It doesn't need to be in a separate file.

# In it, there is an on_epoch_end function, which gets called by 
# the callback whenever the epoch ends. logs objects are also sent 
# containing great information about the updated state of training.

# The code above checks if the loss is less than 0.4 and canceling
# the training itself. In the model.fit, the callbacks parameter is
# used.


# To apply:
# Instantiate the class. Then, in the model.fit use the callbacks
# parameter and pass it to the instance of the class.

# Control training
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images/255.0
test_images = test_images/255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten()
    tf.keras.layers.Dense(512, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs = 5, callbacks = [callbacks])