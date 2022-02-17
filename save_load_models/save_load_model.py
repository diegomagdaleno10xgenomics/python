# save_load_model.py
# Save and load models
# Source: https://www.tensorflow.org/tutorials/keras/save_and_load

import os
import tensorflow as tf
from tensorflow import keras

# Model progress can be saved during and after training. This means a
# model can resume where it left off and avoid long training times.
# Saving also means you can share your model and others can recreate
# your work. When publishing research models and techniques, most
# machine learning practitioners share:
# 1) code to create the model
# 2) the trained weights, or parameters, for the model
# Sharing this data helps others understand how the model works and try
# it themselves with new data.

# Options
# There are different ways to save Tensorflow models depending on the
# API you're using. This guide uses tf.keras, a high-level API to build
# and train models in Tensorflow. For other approaches see the Tensorflow
# Save and Restore guide or Saving in eager.

print(tf.version.VERSION)


# Get an example dataset
# To demonstrate how to save and load weights, you'll use the MNIST
# dataset. To speed up these runs, use the first 1000 examples.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# Define a model
# Start by building a simple sequential model.
# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()


# Save Checkpoints during training
# You can use a trained model without having to retrain it, or pick-up
# training where you left off in case the training process was
# interrupted. The tf.keras.callbacks.ModelCheckpoint callback allows
# you to continually save the model both during and a the end end of
# training.

# Checkpoint callback usage
# Create a tf.keras.callbacks. ModelCheckpoint callback that saves the
# weights only during training.
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

# This creates a single collection of Tensorflow checkpoint files that
# are updated at the end of each epoch.
print(os.listdir(checkpoint_dir))

# As long as two models share the same architecture you can share weights
# between them. So, when restoring a model from weights-only, create a
# model with the same architecture as the original model and then set
# its weights.
# Now, rebuild a fresh, untrained model and evaluate it on the test set.
# An untrained model will perform at chance levels (~10% accuracy):

# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Then load the weights from the checkpoint and re-evaluate.

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# Checkpoint callback options
# The callback provides several options to provide unique names for
# checkpoiints and adjust the checkpointing frequency.
# Train a new model, and save uniquely named checkpoints once every
# five epochs.

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# Now, look at the resulting checkpoints and choose the latest one.
print(os.listdir(checkpoint_dir))

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# To test, reset the model and load the latest checkpoint.
# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# What are these files?
# The above code stores the weights to a collection of checkpoint
# formatted files that contain only the trained weights in a binary
# format. Checkpoints contain:
# 1) One or more shards that contain your model's weights
# 2) An index file that indicates which weights are stored in which 
#   shard.
# If you are training a model on a single machine, you'll have one
# shard with the suffix .data-00000-of-0001


# Manually save weights
# Manually saving weights with the Model.save_weights method. By
# default, tf.keras - and save_weights in particular - uses the
# Tensorflow checkpoint format with a .ckpt extension (saving in HDF5
# with a .h5 extension is covered in the Save and serialize models
# guide).
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# Save the entire model
# Call model.save to save a model's architecture, weights, and training
# configuration in a single file/folder. This allows you to export a
# model so it can be used without access to the original Python code.
# Since the optimizer-state is recovered, you can resume training from
# exactly where you left off.
# An entire model can be saved in two different file formats
# (SavedModel and HDF5). The Tensorflow SavedModel format is the default
# file format in TF2.x. However, models can be saved in HDF5 format.
# More details on saving entire models in the two file formats is
# described below.
# Saving a fully-functional model is very useful - you can load them in
# Tensorflow.js (SavedModel, HDF5) and then train and run them in web
# browsers, or convert them to run on mobile devices using Tensorflow
# Lite (SavedModel, HDF5).
# Note: custom objects (e.g. subclassed models or layers) require
# special attention when saving and loading. See Saving custom objects
# section below.

# SavedModel format
# The SavedModel format is another way to serialize models. Models saved
# in this format can be restored using tf.keras.models.load_model and
# are compatible with Tensorflow Serving. The SavedModel guide goes into
# detail about how to serve/inspect the SavedModel. The section below
# illustrates the steps to save and restore the model.
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
os.makedirs("saved_model", exist_ok=True)
model.save('saved_model/my_model')

# The SavedModel format is a directory containing a protobuf binary and
# a Tensorflow checkpoint. Inspect the saved model directory.

print(os.listdir("saved_model"))
print(os.listdir("saved_model/my_model"))

# Reload a fresh Keras model from the saved model.
new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# The restored model is compiled with the same arguments as the original
# model. Try running and predict with the loaded model.
# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)


# HDF5 format
# Keras provides a basic save format using the HDF5 standard.
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

# Now recreate the model from that file.
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

# Check its accuracy.
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# Keras saves models by inspecting their architectures. This technique
# saves everything:
# 1) The weight values
# 2) The model's architecture
# 3) The model's training configuration (what you pass to the .compile()
#   method)
# 4) The optimizer and its state, if any (this enables you to restart 
#   training where you left off)
# Keras is not able to save the v1.x optimizers (from tf.compat.v1.train)
# since they aren't compatible with checkpoints. For v1.x optimizers,
# you need to re-compile the model after loading - losing the state of
# the optimizer.


# Saving custom objects
# If you are using the SavedModel format, you can skip this section.
# The key difference between HDF5 and SavedModel is that HDF5 uses
# object configs to save the model architecture, while SavedModel saves
# the execution graph. Thus, SavedModels are able to save custom objects
# like subclassed models and custom layers without requiring the original
# code.
# To save custom objects to HDF5, you must do the following:
# 1) Define a get_config method in your object, and optionally from a
#   grom_config classmethod.
#   -> get_config(self) returns a JSON-serializable dictionary of
#       parameters needed to recreate the object.
#   -> from_config(cls, config) uses the returned config from get_config
#       to create a new object. By default, this function will use the 
#       config as initialization kwargs (return cls(**config)).
# 2) Pass the object to the custom_objects argument when loading the
#   model. The argument must be a dictionary mapping the string class
#   name to the Python class. E.g. tf.keras.models.load_model(path,
#   custom_objects={"CustomLayer": CustomLayer}).
# See the Writing layers and models from scratch tutorial for examples
# of custom objects and get_config.