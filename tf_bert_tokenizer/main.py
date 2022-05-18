# main.py
# Load a pre-trained BERT tokenizer. Apply tokenizer (tokenize) to
# vocabulary or text not seen in BERT.
# Source (Tensorflow Text Tokenizers): https://www.tensorflow.org/text/
# guide/tokenizers#subword_tokenizers
# Source (Tensorflow Fine-tune BERT Example):
# https://www.tensorflow.org/text/tutorials/fine_tune_bert
# Tensorflow 2.7.0
# Python 3.7


import os
import tensorflow as tf
import tensorflow_text as text


def main():
	# Pretrained BERT config, vocab, & pre-trained checkpoint.
	gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
	print(tf.io.gfile.listdir(gs_folder_bert))

	# Pretrained BERT encoder.
	hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

	# The BERT Tokenizer
	# To fine-tune a pre-trained model you need to be sure that
	# you're using exactly the same tokenization, vocabulary, and
	# index mapping as you used during training.
	# The BERT tokenizer used in this tutorial is written in pure
	# Python (It's not built out of Tensorflow ops). So you can't
	# just plug it into your model as a keras.layer like you can
	# with preprocessing.TextVectorization.
	# The Following code rebuilds the tokenizer that was used by
	# the base model.
	tokenizer = text.BertTokenizer(
		os.path.join(gs_folder_bert, "vocab.txt"),
		#token_out_type=tf.string, # prints out as string, default is int
		lower_case=True # lowercase text. Will cause UNK tokens to appear for uncased BERT models
	)
	sample = "Hello there Gorn. We welcome you to the Umbaqi Council."
	tokens = tokenizer.tokenize([sample])

	# Should be exactly the same.
	print(tokens.to_list())
	print(tokens)

	# Undo the tokenization.
	untokenized = tokenizer.detokenize(tokens)
	print(untokenized)

	# Exit the program.
	exit()


if __name__ == '__main__':
	main()
