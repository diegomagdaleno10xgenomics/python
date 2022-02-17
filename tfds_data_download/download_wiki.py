# download_wiki.py
# author: Diego Magdaleno
# This program downloads the following wiki datasets from tfds to the current directory.
# Note that once the dataset has been downloaded, it will be loaded from the specified
# directory automatically.
# Documentation on using tfds.load():
# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/api_docs/python/tfds/load
# Documentation on the wiki datasets:
# (wiki40b) https://www.tensorflow.org/datasets/catalog/wiki40b
# (wikipedia) https://www.tensorflow.org/datasets/catalog/wikipedia
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import tensorflow_datasets as tfds


# Download and load wiki datasets. Download is saved to current directory.
x = tfds.load("wiki40b", data_dir=".")
y = tfds.load("wikipedia/20201201.en", data_dir=".")
