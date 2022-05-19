# bert_model.py
# Initialize a BERT model from Tensorflow-Hub and pass sentences
# through it for it's pooled_output embedding.
# Source (TF BERT-Experts Example): https://www.tensorflow.org/hub/tutorials/bert_experts
# Source (TF-Hub BERT Text Embedding): https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4 
# Source (Serialize TF Model): https://www.tensorflow.org/guide/keras/save_and_serialize
# Tensorflow 2.7.0
# Windows/MacOS/Linux


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text


def main():
	# Note: The following code is created in the following Tensorflow
	# versions:
	# Tensorflow 2.7.1 
	# Tensorflow-Hub 0.12.0
	# Tensorflow-Text 2.7.3

	# TF-Hub urls.
	PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
	BERT_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

	# Create the BERT model using the Keras Functional API.
	text_input = keras.layers.Input(shape=(), dtype=tf.string)
	preprocessor = hub.KerasLayer(PREPROCESSOR)
	encoder_inputs = preprocessor(text_input)
	encoder = hub.KerasLayer(BERT_MODEL, trainable=False)
	outputs = encoder(encoder_inputs)
	pooled_output = outputs["pooled_output"] # [batch_size, 768]
	sequence_output = outputs["sequence_output"] # [batch_size, seq_len, 768]
	
	emb_bert = keras.Model(inputs=text_input, outputs=pooled_output, name="emb_bert")
	emb_bert.trainable = False
	print(emb_bert.summary())
	
	# Save/load the model.
	save = "./Embedding_BERT_Model"
	if os.path.exists(save):
		emb_bert = load_model(save)
	else:
		emb_bert.save(save)

	# Sample input to model. Print model output.
	sentences = tf.constant(["Hello there my friend"])
	print(emb_bert(sentences))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
