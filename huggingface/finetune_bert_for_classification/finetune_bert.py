# finetune_bert.py
# Use the huggingface transformers library to fine-tune BERT and other
# transformer models for text classification in python.
# Source: https://www.thepythoncode.com/article/finetuning-bert-using-
# huggingface-transformers-python
# Windows/MacOS/Linux
# Python 3.7


import random
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments


def main():
	# Make a function to set seed for reproducable behavior.
	def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		if is_torch_available():
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed) # Safe to call even if cuda is not available.
		if is_tf_available():
			import tensorflow as tf


	set_seed(1)

	# We'll be using BERT (the bert-base-uncased) pre-trained model. We
	# will also be using a max_length of 512. max_length is the maximum
	# length of our sequence. In other words, we'll be picking only the
	# first 512 tokens from each document or post. You can always change
	# it to whatever you want. However, if you increase it, make sure it
	# fits your memory during the training event when using a lower batch
	# size.
	cache_dir = "./vanilla_bert_base_uncased"
	model_name = "bert-base-uncased"
	max_length = 512

	# Loading the dataset
	# Download and load the tokenizer responsible for converting our text
	# to sequences of tokens:
	tokenizer = BertTokenizerFast.from_pretrained(
		model_name, do_lower_case=True, cache_dir=cache_dir
	)

	# We also set do_lower_case to True to make sure we lowercase all the
	# text (remember, we're using the uncased model).


	def read_20newsgroups(test_size=0.2):
		# Download & load the 20newsgroups datasets from sklearn's repos.
		dataset = fetch_20newsgroups(
			subset="all", shuffle=True, remove=("headers", "footers", "quotes")
		)
		documents = dataset.data
		labels = dataset.target

		# Split into training & testing and return the data as well as the
		# label names.
		return (
			train_test_split(documents, labels, test_size=test_size),
			dataset.target_names
		)


	# Call the function.
	(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()

	# Each of the train_texts and valid_texts is a list of documents (list
	# of strings) for training and validation respectively. The same for
	# train_labels and valid_labels, each of them is a list of integers,
	# or labels ranging from 0 to 19. target_names is a list of our 20
	# labels, each has its own name.

	# Tokenize (encode) the corpus. We set truncation to True so that we 
	# eliminate tokens that go above max_length and also set padding to	
	# True to pad documents that are less than max_length with empty
	# tokens.
	train_encodings = tokenizer(
		train_texts, truncation=True, padding=True, max_length=max_length
	)
	valid_encodings = tokenizer(
		valid_texts, truncation=True, padding=True, max_length=max_length
	)


	# Wrap the tokenized text data into a pytorch Dataset.
	class NewsGroupsDataset(torch.utils.data.Dataset):
		def __init__(self, encodings, labels):
			self.encodings = encodings
			self.labels = labels


		def __getitem__(self, idx):
			item = {k:torch.tensor(v[idx]) for k, v in self.encodings.items()}
			item["labels"] = torch.tensor([self.labels[idx]])
			return item


		def __len__(self):
			return len(self.labels)

	
	# Convert the tokenized data into torch Dataset
	train_dataset = NewsGroupsDataset(train_encodings, train_labels)
	valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

	# Train the model
	# Now that the data is prepared, download and load the BERT model and
	# its pre-trained weights. We're using the
	# BertForSequenceClassification class from the huggingface
	# transformers library. We set num_labels to the length of the
	# available labels (target_names), which in this case is 20.
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = BertForSequenceClassification.from_pretrained(
		model_name, num_labels=len(target_names), cache_dir=cache_dir
	).to(device)


	# Make a simple function to compute the metrics we want (accuracy in
	# this case). Feel free to include any metric you want.
	def compute_metrics(pred):
		labels = pred.label_ids
		preds = pred.predictions.argmax(-1)

		# Calculate accuracy using sklearn's function.
		acc = accuracy_score(labels, preds)
		return {
			"accuracy": acc,
		}


	# Use the TrainingArguments class to specify the training arguments,
	# such as the number of epochs, batch size, and some other parameters.
	training_args = TrainingArguments(
		output_dir="./results",		# output directory
		num_train_epochs=3,		# total number of training epochs
		per_device_train_batch_size=8,	# batch size per device during training
		per_device_eval_batch_size=20,	# batch size for evaluation
		warmup_steps=500,		# number of warmup steps for learning rate scheduler
		weight_decay=0.01,		# strength of weight decay
		logging_dir="./logs",		# direectory for storing logs
		load_best_model_at_end=True,	# load the best model when finished training (default metric is loss) but you can specify 'metric_for_best_model' argument to change to accuracy or other metrics
		logging_steps=400,		# log & save weights at each logging_steps
		save_steps=400,			# 
		evaluation_strategy="steps",	# evaluate each 'logging_steps'
	)

	# Pass the training arguments, dataset, and compute_metrics callback
	# to the trainer.
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
		compute_metrics=compute_metrics,
	)

	# Train the model.
	trainer.train()

	# Evaluate the current model after training.
	trainer.evaluate()

	# Save teh fine-tuned model & tokenizer.
	model_path = "./20newsgroups-bert-base-uncased"
	model.save_pretrained(model_path)
	tokenizer.save_pretrained(model_path)

	# Performing inference
	

	# This function takes a text as a string, tokenizes it with the
	# tokenizer, calculates the output probabilities using the softmax
	# function, and returns the actual label.
	def get_prediction(text):
		# Prepare the text into tokenized sequence.
		inputs = tokenizer(
			text, padding=True, truncation=True, max_length=max_length, 
			return_tensors="pt"
		)

		# Perform inference on the model.
		outputs = model(**inputs)

		# Get output probabilities by doing softmax.
		probs = outputs[0].softmax(1)
		
		# Executing argmax functions to get the candidate label.
		return target_names[probs.argmax()]


	# Inference examples.
	text = """
The first thing is first. 
If you purchase a Macbook, you should not encounter performance issues that will prevent you from learning to code efficiently.
However, in the off chance that you have to deal with a slow computer, you will need to make some adjustments. 
Having too many background apps running in the background is one of the most common causes. 
The same can be said about a lack of drive storage. 
For that, it helps if you uninstall xcode and other unnecessary applications, as well as temporary system junk like caches and old backups.
"""
	print("Example 1:")
	print(text)
	print(get_prediction(text))
	print("=" * 50)

	text = """
A black hole is a place in space where gravity pulls so much that even light can not get out. 
The gravity is so strong because matter has been squeezed into a tiny space. This can happen when a star is dying.
Because no light can get out, people can't see black holes. 
They are invisible. Space telescopes with special tools can help find black holes. 
The special tools can see how stars that are very close to black holes act differently than other stars.
"""
	print("Example 2:")
	print(text)
	print(get_prediction(text))
	print("=" * 50)

	text = """
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.
Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  
Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
"""
	print("Example 3:")
	print(text)
	print(get_prediction(text))
	print("=" * 50)

	# Conclusion
	# You can also use other transformer model such as GPT-2 with
	# GPT2ForSequenceClassification, RoBERTa with
	# GPT2ForSequenceClassification, DistilBERT with
	# DistilBERTForSequenceClassification, and much more. 

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
