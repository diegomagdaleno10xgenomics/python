# train_distilbert.py
# Train a distilBERT (or BERT small) model from scratch on Esperanto
# using the huggingface tokenizers and transformers libraries.
# Source: https://huggingface.co/blog/how-to-train
# Source (Colab): https://colab.research.google.com/github/huggingface/
#	blog/blob/master/notebooks/01_how_to_train.ipynb
# Windows/MacOS/Linux
# Python 3.7


from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import pipeline
from transformers import TokenClassificationPipeline
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling


def main():
	# This example will train a "small" model (84M parameters => 6 layers,
	# 768 hidden size, 12 attention heads) with the same number of layers
	# and heads as DistilBERT on Esperanto.
	# Esperanto is a constructed language with a goal of being easy to
	# learn. It was selected for several reasons, including:
	# -> it is a relatively low-resource (even though it's spoken by ~2M
	#	people).
	# -> its grammar is highly regular (eg all common nouns end in -o, all
	#	adjectives end in -a) so we should get interesting linguistic 
	#	results even on a small dataset.
	# -> the overarching goal at the foundation of the language is to
	#	bring people closer (fostering world peace & international
	#	understanding).
	# The model will be called EsperBERTo.

	# The dataset
	# Here, we'll use the Esperanto protion of the OSCAR corpus from
	# INRIA. OSCAR is a huge multilingual corpus obtained by language
	# classification and filtering of Common Crawl dumps of the web. The
	# Esperanto portion of the dataset is only 299M, so we'll concatenate
	# with the Esperanto sub-corpus of the Leipzig Corpora Collection,
	# wich is comprised of text from diverse sources like news,
	# literature, and wikipedia. The final training corpus has a size of
	# 3GB, which is still small -> for your model, you will get better
	# results with more data you can get to pre-train on.
	# Here is one of the files (the Oscar one). You can download it by:
	# wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt
	
	# Train the tokenizer
	# We choose to train a byte-level Byte-pair encoding tokenizer (the
	# same as GPT-2), with the same special tokens as RoBERTa. Let's pick
	# its arbitrary size to be 52,000. We recommend training a byte-level 
	# BPE (rather than let's say, a WordPiece tokenizer like BERT) because
	# it will start building its vocabulary from an alphabet of single
	# bytes, so all words will be decomposable into tokens (no more <unk>
	# tokens).
	paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")

	# Initialize a tokenizer.
	tokenizer = ByteLevelBPETokenizer()

	# Custom training.
	tokenizer.train(
		files=paths, vocab_size=52_000, min_frequency=2, 
		special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
	)

	# Save files to disk.
	tokenizer.save_model(".", "esperberto")

	# Now we have both a vocab.json which is a list of the most frequent
	# tokens ranked by frequency, and a merges.txt list of merges.
	# What is great is that our tokenizer is optimized for Esperanto.
	# Compared to a generic tokenizer trained for English, more native
	# words are represented by a single, unsplit token. Diacritics, ie
	# accented characters used in Esperanto are encoded natively. We also
	# represent sequences in a more efficient manner. Here on this corpus,
	# the average length of the encoded sequences is ~30% smaller than
	# using the pre-trained GPT-2 tokenizer.
	# Here's how you can use it in tokenizers, including handling the
	# RoBERTa. You'll also be able to use it directly from transformers.
	#from tokenizers.implementations import ByteLevelBPETokenizer
	#from tokenizers.processors import BertProcessing
	#tokenizer = ByteLevelBPETokenizer(
	#	"./models/EsperBERTo-small/vocab.json",
	#	"./models/EsperBERTo-small/merges.txt",
	#)
	#tokenizer._tokenizer.post_processor = BertProcessing(
	#	("</s>", tokenizer.token_to_id("</s>)),
	#	("<s>", tokenizer.token_to_id("<s>")),
	#)
	#tokenizer.enable_truncation(max_length=512)
	#print(
	#	tokenizer.encode("Mi estas Julien.")
	#)

	# Train the language model from scratch
	# Update: The associated Colab notebook uses the Trainer directly, 
	# instead of through a script.
	# We will now train the new language model using the Trainer class.
	dataset = LineByLineTextDataset(
		tokenizer=tokenizer,
		file_path="./oscar.eo.txt",
		block_size=128,
	)
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=True,
		mlm_probability=0.15
	)
	training_args = TrainingArguments(
		output_dir="./EsperBERTo",
		overwrite_output_dir=True,
		num_train_epochs=1,
		per_gpu_train_batch_size=64,
		save_steps=10_000,
		save_total_limit=2,
		prediction_loss_only=True,
	)
	trainer = Trainer(
		model=model,
		args=training_args
		data_collator=data_collator,
		train_dataset=dataset,
	)
	trainer.train()
	trainer.save_model("./models/EsperBERTo-small")

	# Check that the language model actually trained
	fill_mask = pipeline(
		"fill-mask", 
		model="./models/EsperBERTo-small",
		tokenizer="./models/EsperBERTo-small",
	)
	result = fill_mask("La suno <mask>") # "The sun <mask>"
	print(result)

	result = fill_mask("Jen la komenco de bel <mask>") # "This is the beginning of a beautiful <mask>"
	print(result)

	# With more complex prompts, you can probe whether your language model
	# captured more semantic knowledge or even some sort of (statistical)
	# common sense reasoning.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
