# enhance.py
# Perform image super resolution using a pretrained ESRGAN model from 
# tensorflow hub. The ESRGAN from tensorflow hub was trained on the 
# div2k dataset (on bicubically downsampled iamges) on image patches of
# size 128 x 128.


import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


def main():
	# Run this command before the program.
	# wget "https://user-images.githubusercontent.com/12981474/40157448-eff91f06-5953-11e8-9a37-f6b5693fa03f.png" -O original.png

	# Declaring constants.
	IMAGE_PATH = "original.png"
	SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


	# Define helper functions
	# Load image from path and preprocess to make it model ready.
	# @param: image_path, path to the image file.
	def preprocess_image(image_path):
		hr_image = tf.image.decode_image(tf.io.read_file(image_path))

		# If png, remove the alpha channel. The model only supports images
		# with 3 color channels.
		if hr_image.shape[-1] == 4:
			hr_image = hr_image[..., :-1]
		hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
		hr_image = tf.image.crop_to_bounding_box(
			hr_image, 0, 0, hr_size[0], hr_size[1]
		)
		hr_image = tf.cast(hr_image, tf.float32)
		return tf.expand_dims(hr_image, 0)


	# Saves unscaled tensor images.
	# @param: image, 3D image tensor (height, width, channels).
	# @param: filename, name of the file to save.
	def save_image(image, filename):
		if not isinstance(image, Image.Image):
			image = tf.clip_by_value(image, 0, 255)
			image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
		image.save("%s.jpg" % filename)
		print("Saved as %s.jpg" % filename)

	
	# Plots images from the image tensors.
	# @param: image, 3D image to tensor (height, width, channels).
	# @param: title, Title to display in the plot.
	def plot_image(image, title=""):
		image = np.asarray(image)
		image = tf.clip_by_value(image, 0, 255)
		image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
		plt.imshow(image)
		plt.axis("off")
		plt.title(title)


	# Perform super resolution of images loaded from path
	hr_image = preprocess_image(IMAGE_PATH)

	# Plotting original resolution image.
	plot_image(tf.squeeze(hr_image), title="Original Image")
	save_image(tf.squeeze(hr_image), filename="Original Image")

	model = hub.load(SAVED_MODEL_PATH)

	start = time.time()
	fake_image = model(hr_image)
	fake_image = tf.squeeze(fake_image)
	print("Time take: %f" % (time.time() - start))

	# Plotting super resolution image.
	plot_image(tf.squeeze(fake_image), title="Super Resolution Image")
	save_image(tf.squeeze(fake_image), filename="Super Resolution Image")

	# Evaluate model performance
	# Run before the program.
	# wget "https://lh4.googleusercontent.com/-Anmw5df4gj0/AAAAAAAAAAI/AAAAAAAAAAc/6HxU8XFLnQE/photo.jpg64" -O test.jpg
	IMAGE_PATH = "test.jpg"


	# More helper functions
	# Scales down images using bicubic downsampling.
	# @param: image, 3D or 4D tensor of preprocessed image.
	def downscale_image(image):
		image_size = []
		if len(image.shape) == 3:
			image_size = [image.shape[1], image.shape[0]]
		else:
			raise ValueError("Dimension mismatch. Can work only on single image")
	
		image = tf.squeeze(
			tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
		)

		lr_image = np.asarray(
			image.nump()\
				.resize([image_size[0] // 4, image_size[1] // 4], Image.BICUBIC)
		)
		lr_image = tf.expand_dims(lr_image, 0)
		lr_image = tf.cast(lr_image, tf.float32)
		return lr_image


	hr_image = preprocess_image(IMAGE_PATH)
	lr_image = downscale_image(tf.squeeze(hr_image))

	# Plotting low resolution image.
	plot_image(tf.squeeze(lr_image), title="Low Resolution")

	model = hub.load(SAVED_MODEL_PATH)

	start = time.time()
	fake_image = model(lr_image)
	fake_image = tf.squeeze(fake_image)
	print("Time take: %f" % (time.time() - start))

	# Plotting super resolution image.
	plot_image(tf.squeeze(fake_image), title="Super Resolution")

	# Calculate PSNR wrt of original image.
	psnr = tf.image.psnr(
		tf.clip_by_value(fake_image, 0, 255),
		tf.clip_by_value(hr_image, 0, 255),
		max_val=255
	)
	print("PSNR Achieved: %f" % psnr)

	# Compare outputs side by side
	plt.rcParams['figure.figsize'] = [15, 10]
	fig, axes = plt.subplots(1, 3)
	fig.tight_layout()
	plt.subplot(131)
	plot_image(tf.squeeze(hr_image), title="Original")
	plt.subplot(132)
	fig.tight_layout()
	plot_image(tf.squeeze(lr_image), "x4 Bicubic")
	plt.subplot(133)
	fig.tight_layout()
	plot_image(tf.squeeze(fake_image), "Super Resolution")
	plt.savefig("ESRGAN_DIV2K.jpg", bbox_inches="tight")
	print("PSNR: %f" % psnr)
	
	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
