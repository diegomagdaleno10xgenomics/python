# train_bert.py
# Train BERT model model from scratch using the cc_news (common crawl
# news) dataset and Huggingface transformers library.
# Source: https://www.thepythoncode.com/article/pretraining-bert-
# huggingface-transformers-in-python
# Windows/MacOS/Linux
# Python 3.7


import os
import json
from datasets import *
from tokenizers import *
from transformers import *


def main():
	# A pre-trained model is a model that was previously trained on a
	# large dataset and saved for direct use or fine-tuning. This tutorial
	# will train BERT (or you can apply this to any other transformer
	# model) from scratch on a custom raw dataset with the help of the
	# huggingface transformer library in Python.
	# Pre-training on transformers can be done with self-supdervised
	# tasks, below are some popular tasks done on BERT:
	# -> Masked Language Modeling (MLM): This task consists of masking a
	#	certain percentage of the tokens in the sentence, and the model is
	#	trained to predict those masked words. We'll be using this one in
	#	this tutorial example.
	# -> Next Sentence Prediction (NSP): The model recieves pairs of
	#	sentences as input and learns to predict whether the second sentence
	#	in the pair is the subsequent sentence in the original document.

	# Picking a dataset
	# If you're willing to pre-train a transformer, then you most likely
	# have a custom dataset. For the demonstration purpose of this
	# tutorial, we're going to use the cc_news dataset, we'll be using
	# huggingface datasets library for that. As a result, make sure to
	# follow this link
	# (https://huggingface.co/docs/datasets/dataset_script.html) to get
	# your custom dataset to be loaded in the library.
	# CC-News dataset contains news articles from news sites around all
	# over the world. It contains 708,241 news articles in English
	# published between January 2017 and December 2019.

	# Cache the dataset and models here in the current working directory.
	cache_dir = "."

	# Download the cc_news dataset.
	dataset = load_dataset("cc_news", split="train", cache_dir=cache_dir)

	# There is only one split in this dataset, so we need to split it into
	# training and testing sets (you can pass the seed parameter to the
	# train_test_split() method so it'll be the same sets after running
	# multiple times):
	d = dataset.train_test_split(test_size=0.1, seed=42) # d["train"], d["test"]
	
	# Let's see how it looks like:
	for t in d["train"]["text"][:3]:
		print(t)
		print("=" * 50)

	# As mentioned previously, if you have your custom datset, you can
	# either follow the above link of setting up our dataset to be loaded
	# as above, or you can use the LineByLineTextDataset class if your
	# custom dataset is a text file where all sentences are separated bu a
	# new line.
	# However, a better way than using LineByLineDataset is to set up your
	# custom dataset by splitting your text file into several chunks using
	# the split command or any other Python code, and load them using
	# load_dataset() as done above:
	#files = ["train1.txt", "train2.txt"] # train3.tct, etc
	#dataset = load_dataset("text", data_files=files, split="train")
	# If you have your custom dataset as one massive file, then you should
	# divide it into a handful of text files (such as using the split 
	# command on Linux or Colab) before loading them using the
	# load_dataset() function, as the runtime will crash if it exceeds the
	# memory.


	# Train the tokenizer
	# Next, we need to train the tokenizer. To do that, we need to write
	# our dataset into text files, as that's what the tokenizers library
	# requires the input to be:
	def dataset_to_text(dataset, output_filename="data.txt"):
		# Utility function to save dataset text to disk. Useful for using the
		# texts to train the tokenizer.
		with open(output_filename, "w") as f:
			for t in dataset["text"]:
				print(t, file=f)


	# If you already have your dataset as text files, then you should skip
	# this step.
	if not os.path.exists("train.txt"):
		# Save the training set to train.txt.
		dataset_to_text(d["train"], "train.txt")
	if not os.path.exists("test.txt"):
		# Save the testing set to test.txt.
		dataset_to_text(d["test"], "test.txt")

	# Next, let's define some parameters.
	special_tokens = [
		"[PAD]", "[INK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
	]

	# If you want to trian the tokenizer on both sets:
	# files = ["train.txt", "test.txt"]
	# Training the tokenizer on the training set.
	files = ["train.txt"]

	# 30,522 vocab is BERT's default vocab size, feel free to tweak.
	vocab_size = 30_522

	# Maximum sequence length, lowering will result in faster training
	# (when increasing batch size).
	max_length = 512

	# Whether to truncate.
	truncate_longer_samples = True
	
	# The files list is the list of tiles to pass to the tokenizer for
	# training. vocab_size is the vocabulary size of tokens. max_length is
	# the maximum sequence length. Now we can train the tokenizer.

	# Initialize the WordPiece tokenizer.
	tokenizer = BertWordPieceTokenizer()

	# Train the tokenizer.
	tokenizer.train(
		files=files, vocab_size=vocab_size, special_tokens=special_tokens
	)

	# Enable truncation up to the maximum 512 tokens.
	tokenizer.enable_truncation(max_length=max_length)

	# Since this is BERT, the default tokenizer is WordPiece. As a result,
	# we initialize the BertWordPieceTokenizer() tokenizer class from the
	# tokenizers library and use the train() method to train it. Let's
	# save it now.
	model_path = "pretrained_bert"

	# make the directory if not already there.
	if not os.path.isdir(model_path):
		os.mkdir(model_path)

	# Save the tokenizer.
	tokenizer.save_model(model_path)

	# Dumping some of the tokenizer config to the config file, including
	# special tokens, whether to lowercase, and the maximum sequence
	# length.
	with open(os.path.join(model_path, "config.json"), "w") as f:
		tokenizer_cfg = {
			"do_lower_case": True,
			"unk_token": "[UNK]",
			"unk_token": "[SEP]",
			"unk_token": "[PAD]",
			"unk_token": "[CLS]",
			"unk_token": "[MASK]",
			"model_max_length": max_length,
			"max_len": max_length,
		}
		json.dump(tokenizer_cfg, f)

	# The tokenizer.save_model() method saves the vocabulary file into
	# that path. We also manually save some tokenizer configurations, such
	# as:
	# unk_token -> A special token that represents out-of-vocabulary
	#	tokens, even though the tokenizer is a WordPiece tokenizer, the unk
	#	tokens are not impossible but rare.
	# sep_token -> A special token that separates two different sentences
	#	in the same input.
	# pad_token -> A special token that is used to fill sentences that do
	#	not reach the maximum sequence length (since the arrays of tokens
	#	must be the same size).
	# cls_token -> A special token representing the class of the input.
	# mask_token -> This is the mask token we use for the Masked Language
	#	Modeling (MLM) pretraining task.
	# After the training of the tokenizer is complete (it should take
	# several minutes), let's load it now:
	tokenzier = BertTokenizerFast.from_pretrained(model_path) # Load it as BertTokenizerFast


	# Tokenizing the dataset
	# Now that we have the tokenizer ready, the below code is responsible
	# for tokenizing the dataset.
	def encode_with_truncation(examples):
		# Mapping function to tokenize the sentences passed with truncation.
		return tokenizer(
			examples["text"], truncation=True, padding="max_length", 
			max_length=max_length, return_special_tokens_mask=True
		)	

	
	def encode_without_truncation(examples):
		# Mapping function to tokenize the sentences passed without truncation.
		return tokenizer(
			examples["text"], return_special_tokens_mask=True
		)	


	# the encode function will depend on the truncate_longer_samples
	# variable.
	encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

	# Tokenize the train dataset.
	train_dataset = d["train"].map(encode, batched=True)
	# Tokenize the testing dataset.
	test_dataset = d["test"].map(encode, batched=True)
	if truncate_longer_samples:
		# Remove other columns and set input_ids and attention_mask.
		train_dataset.set_format(
			type="torch", columns=["input_ids", "attention_mask"]
		)
		test_dataset.set_format(
			type="torch", columns=["input_ids", "attention_mask"]
		)
	else:
		train_dataset.set_format(
			columns=["input_ids", "attention_mask", "special_tokens_mask"]
		)
		test_dataset.set_format(
			columns=["input_ids", "attention_mask", "special_tokens_mask"]
		)

	# The encode callback that is used to tokenize the dataset depends on
	# the truncate_longer_samples boolean variable. If set to True, then
	# the sentences that exceed the maximum sequence length (max_length
	# parameter) are truncated. Otherwise, they aren't.
	# Next, in the case of truncate_longer_samples being False, we need to
	# join the untruncated samples together and cut them into fixed-size
	# vectors since the model expects a fixed size sequence during
	# training.

	
	# Main data processing function that will concatenate all texts from
	# the dataset and generate chunks of max_seq_length.
	def group_texts(examples):
		# Concatenate all texts.
		concatenated_examples = {k:sum(examples[k], []) for k in examples.keys()}
		total_length = len(cocatenated_examples[list(examples.keys())[0]])

		# Drop the small remainder. We could add padding if the model
		# supported it instead of this drop. You can customize this part to
		# your needs.
		if total_length >= max_length:
			total_length = (total_length // max_length) * max_length

		# Split by chunks of max_len.
		result = {
			k:[t[i:i + max_length] for i in range(0, total_length, max_length)]
			for k,t in concatenated_examples.items()
		}
		return result


	# Note that with batched=True, this map processes 1,000 texts
	# together, so group_texts throws away a remainder for each of those
	# groups of 1,00 texts. You can adjust that batch_size here but a
	# higher value might be slower to preprocess.
	# To speed this up, use multiprocessing. See the documentation of the 
	# map method for more information:
	# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
	if not truncate_longer_samples:
		train_dataset = train_dataset.map(
			group_texts, batch=True, batch_size=2_000,
			desc=f"Grouping texts in chunks of {max_length}"
		)
		test_dataset = test_dataset.map(
			group_texts, batch=True, batch_size=2_000,
			desc=f"Grouping texts in chunks of {max_length}"
		)

	# Most of the code above was brought from the run_mlm.py script from
	# huggingface transformers examples
	# (https://github.com/huggingface/transformers/tree/master/examples),
	# so this is actually used by the library itself.
	# If you dont want to concatenate all texts and then split them into
	# chunks of 512 tokens, then make sure to set truncate_longer_samples
	# to True, so it will treat each line as an individual sample
	# regardless of its length. Note that if set truncate_longer_samples
	# to True, the above code wont be executed at all.

	# Loading the model
	# For this tutorial, we are using BERT, but feel free to pick any of
	# the transformer models by huggingface transformers library, such as
	# RobertaForMaskedLM or DistilBertForMaskedLM.

	# Initialize the mdoel with the config.
	model_config = BertConfig(
		vocab_size=vocab_size, max_position_embeddings=max_length
	)
	model = BertForMaskedLM(config=model_config)

	# Initialize the model config using BertConfig, and pass the
	# vocabulary size as well as the maximum sequence length. Then pass
	# the config to BertForMaskedLM to initialize the model itself.

	# Pre-training
	# Before we start pre-training the model, we need a way to randomly
	# mask tokens in the dataset for the Masked Language Model (MLM) task.
	# Luckily, the library makes this easy by simply constructing a
	# DataCollatorForLanguageModeling object:
	
	# Initialize the data collator, randomly masking 20% (default is 15%)
	# of the tokens for the Masked Language Modelig (MLM) task.
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer, mlm=True, mlm_probability=0.2
	)

	# Pass the tokenizer and set mlm to True, and also set the
	# mlm_probability to 0.2 to randomly replace each token with [MASK]
	# token by 20% probability. Next, initialize the training arguments:
	training_args = TrainingArguments(
		output_dir=model_path,		# output directory to where save model checkpoint
		evaluation_strategy="steps",	# evaluate each 'logging_steps' steps
		overwrite_output_dir=True,
		num_train_epochs=10,		# number of training epochs, feel free to tweak
		per_device_train_batch_size=10,	# the training batch size, put it as high as your GPU memory fits
		gradient_accumulation_steps=8,	# accumulating the gradients before updating the weights
		per_device_eval_batch_size=64,	# evaluation batch size
		logging_steps=500,
		save_steps=500,
		#load_best_model_at_end=True,	# whether to load the best model (in terms of loss) at the end of training
		#save_total_limit=3,		# whether you don't have much space so you let only 3 model weights saved in the disk
	)

	# Each argument is explained in the comments, refer to the
	# TrainingArguments docs
	# (https://huggingface.co/docs/transformers/main_classes/trainer#trainingarguments)
	# for more details. Now let's make the trainer:

	# Initialize the trainer and pass everything to it.
	trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
	)

	# Pass the training arguments to the Trainer, as well as the model, 
	# data collator, and the training sets. Simply call train() now to
	# start training:

	# Train the model.
	trainer.train()

	# The training will take several hours to several days, depending on
	# the dataset size, training batch size (ie increase it as much as your
	# GPU memory fits), and GPU speed.
	# As you can see in the output, the model is still improving and the
	# validation loss is still decreasing. You usually have to cancel the
	# training once the validation loss stops decreasing.
	# Since we have set logging_steps and save_steps to 1000, then the
	# trainer will evaluate and save the model after every 1000 steps (ie
	# trained on steps x gradient_accumulation_step x
	# per_device_train_size = 1000 x 8 x 10 = 80,000 samples). As a
	# result, I had canceled the training after about 19 hours of
	# training, or 10000 steps (that is about 1.27 epochs, or trained on
	# 800,000 samples), and started to use the model. In the next section,
	# we'll see how we can use the model for inference.

	# Using the model
	# Before using the model, assume that we don't have model and
	# tokenizer variables in the current runtime. Therefore, they need to
	# be loaded again.

	# Load the model checkpoint.
	model = BertForMaskedLM.from_pretrained(
		os.path.join(model_path, "checkpoint-10000")
	)
	# Load the tokenizer.
	tokenizer = BertTokenizerFast.from_pretrained(model_path)

	# Let's use the mdoel now:
	fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

	# Use the simple pipeline API, and pass both the model and the
	# tokenizer. Let's predict some examples:
	examples = [
		"Today's most trending hashtags on [MASK] is Donald Trump",
		"The [MASK] was cloudy yesterday, but today it's rainy."
	]
	for example in examples:
		for prediction in fill_mask(example):
			print(f"{prediction['sequence']}, confidence: {prediction['score']}")
		print("="*50)

	# Conclusion
	# And there you have it. The complete code for pre-training BERT or
	# other transformers using huggingface libraries. Below are some tips:
	# -> As mentioned above, the training speed will depend on the GPU 
	#	speed, the number of the samples in the dataset, and batch size. I
	#	have set the training batch size to 10, as that's the maximum it can
	#	fit my GPU memory on Colab. If you have more memory, make sure to
	#	increase it so you can increase the training speed significantly.
	# -> During training, if you see hte validation loss start to increase,
	#	make sure to remember the checkpoint where the lowest validation
	#	loss occurs so you can load that cehckpoint later for use. You can
	#	also set load_best_model_at_end to True if you don't want to keep
	#	track of the loss, as it will load the best weights in terms of loss
	#	when the training ends.
	# -> The vocabulary size was chosen based on the original BERT 
	#	configuration, as it had the size of 30,522, feel free to increase
	#	it if you feel the language of your dataset has a large vocabulary,
	#	or your can experiment with this.
	# -> If you set truncate_longer_samples to False, then the code 
	#	assumes you have larger text on one sentence (ie line), you will
	#	feel that it takes much longer to process, especially if you set a
	#	larger batch_size on the map() method. If it takes a lot of hours
	#	to process, then you can either set truncate_longer_samples to True
	#	so you truncate sentences athat exceed max_length tokens or you can
	#	save the dataset after processing using the save_to_disk() method,
	#	so you process it once and load it several times.
	# If you're interested in fine-tuning BERT for a downstream task such
	# as text classification, then this tutorial
	# (https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python)
	# will guide you through it.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
