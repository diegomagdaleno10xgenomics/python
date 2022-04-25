# distlbert_qa.py
# Source: https://blog.tensorflow.org/2020/05/how-hugging-face-achieved-2x-
# performance-boost-question-answering.html
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering


def main():
	# Initialize Huggingface transformers model. Be sure to use the
	# Tensorflow implementation provided by the library (has "TF"
	# prepended to the model name. ie DistilBertForQuestionAnswering vs
	# TFDistilBertForQuestionAnswerting).
	distilbert = TFDistilBertForQuestionAnswering.from_pretrained(
		"distilbert-base-cased-distilled-squad",
		cache_dir="./huggingface-distilbert-pretrained"
	)

	# Pass the TF/Keras model's call function to tf.function. This will
	# create a callable, which will be used to trace the call function
	# with a specific signature and shape thanks to get_concrete_function.
	callable = tf.function(distilbert.call)

	# By calling get_concrete_function, we trace-compile the Tensorflow
	# operations of the model for an input signature of two tensors of
	# shape [None, 384], the first one beingthe input ids and the second
	# one the attention_mask.
	concrete_function = callable.get_concrete_function(
		[
			tf.TensorSpec([None, 384], tf.int32, name="input_ids"),
			tf.TensorSpec([None, 384], tf.int32, name="attention_mask"),
		]
	)

	# Save the model in SavedModel format.
	tf.saved_model.save(
		distilbert, 
		"tf_distilbert_cased_savedmodel", 
		signatures=concrete_function,
	)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
