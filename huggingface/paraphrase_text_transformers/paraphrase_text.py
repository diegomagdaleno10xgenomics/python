# paraphrase_text.py
# Explore the different pre-trained transformer models in the
# transformers library to paraphrase sentences in python.
# Source: https://www.thepythoncode.com/article/paraphrase-text-using-
# transformers-in-python
# Windows/MacOS/Linux
# Python 3.7


from transformers import *


def main():
	# Paraphrasing is the process of coming up with someone else's idea in
	# your own words. To paraphrase a text, you have to re-write it
	# without changing its meaning. This example will explore different
	# pre-trained transformer models for automatically paraphrasing text
	# using the huggingface transformers library in python.

	# Pegasus transformer
	# We'll use the Pegasus transformer architecture model that was 
	# fine-tuned for paraphrasing instead of summarization. To instantiate
	# the model, use the PegasusForConditionalGeneration since it is a
	# form of text generation.
	cache_dir = "./pegasus_paraphrase"
	model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase", cache_dir=cache_dir)
	tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase", cache_dir=cache_dir)


	# Create a general function that takes a model, its tokenizer, the
	# target sentence and returns the paraphrased text.
	def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
		# Tokenize the text to be the form of a list of token IDs.
		inputs = tokenizer(
			[sentence], truncation=True, padding="longest", return_tensors="pt"
		)

		# Generate the paraphrased sentences.
		outputs = model.generate(
			**inputs,
			num_beams=num_beams,
			num_return_sequences=num_return_sequences,
		)

		# Decode the generated sentences using the tokenizer to get them back
		# to text.
		return tokenizer.batch_decode(outputs, skip_special_tokens=True)


	# We add the possibility of generating multiple paraphrased sentences
	# by passing num_return_sequences to the model.generate() method. We
	# also set num_beams so we generate the paraphrasing using beam
	# search. Setting it to 5 will allow the model to look ahead for five
	# possible words to keep the most likely hyposthesis at each time step
	# and choose the one that has the overall highest probability.

	# Let's use the function now.
	sentence = "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences."
	paraphrased_sentences = get_paraphrased_sentences(
		model,
		tokenizer,
		sentence,
		num_beams=10,
		num_return_sequences=10
	)
	print("Paraphrasing using Pegasus")
	print("Original sentence:", sentence)
	print("-" * 50)
	for sentence in paraphrased_sentences:
		print(sentence)
	print("=" * 50)

	# T5 transformer
	# This section will explore the T5 architecture model that was
	# fine-tuned on the PAWS dataset. PAWS consists of 108,463
	# human-labeled and 656k noisily labeled pairs. 

	# Load the model and tokenizer.
	cache_dir = "./t5_paraphrase"
	tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", cache_dir=cache_dir)
	model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws", cache_dir=cache_dir)

	# Let's reuse the function defined above and change the input sentence
	# a bit.
	sentence = "One of the best ways to learn is to teach what you've already learned."
	paraphrased_sentences = get_paraphrased_sentences(
		model,
		tokenizer,
		sentence,
	)
	print("Paraphrasing using T5 transformer")
	print("Original sentence:", sentence)
	print("-" * 50)
	for sentence in paraphrased_sentences:
		print(sentence)
	print("=" * 50)

	# These are promising results too. However, if you get some
	# not-so-good paraphrased text, you can prepend the input text with
	# "paraphrase: ", as T5 was intended for multiple text-to-text NLP
	# tasks such as machine translation, text summarization, and more. It
	# was pre-trained and fine-tuned like that.


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
