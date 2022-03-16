# biggan.py
# Download the biggan model from tensorflow hub and run it.
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


tf.compat.v1.disable_eager_execution() # Disable eager execution (default for TF 2)


def main():
	# Load BigGAN-deep 256 module.'
	module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-256/1')
	
	# Sample random noise (z) and ImageNet label (y) inputs.
	batch_size = 8
	truncation = 0.5 # scalar truncation value is [0.0, 1.0]
	z = truncation * tf.random.truncated_normal([batch_size, 128]) # noise sample
	y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
	y = tf.one_hot(y_index, 1000) # One-hot ImageNet label

	# Call BigGAN on a dict of the inputs to generate a batch of images
	# with shape [8, 256, 256, 3] and range [-1, 1].
	samples = module(dict(y=y, z=z, truncation=truncation))
	print(samples)

	tf.compat.v1.summary.image("pics", samples, max_outputs=batch_size)

	init = tf.compat.v1.global_variables_initializer()
	summary = tf.compat.v1.summary.merge_all()
	
	with tf.compat.v1.Session() as sess:
		sess.run(init)
		
		summ = sess.run(summary)
		
		# Create a writer for the summary.
		summary_writer = tf.compat.v1.summary.FileWriter(
			"./tmp/biggan/", sess.graph
		)
		summary_writer.add_summary(summ, 1)
		summary_writer.flush()
		summary_writer.close()

	# To view the results, open tensorboard and point it to the summary
	# directory.
	# "tensorboard --logdir=./tmp/biggan/"

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
