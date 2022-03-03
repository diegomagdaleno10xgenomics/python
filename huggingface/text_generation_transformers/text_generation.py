# text_generation.py
# Generate any type of text with GPT-2 and GPT-J transformer models
# with the huggingface transformers library in python.
# Source: https://www.thepythoncode.com/article/text-generation-with-
# transformers-in-python
# Windows/MacOS/Linux
# Python 3.7


from transformers import pipeline


def main():
	# Text generation is the task of automatically generating text using
	# machine learning so that it cannot be distinguished between whether
	# it's written by a human or a machine. It is also widely used for
	# text suggestion and completion in various real-word applications. In
	# recent years, a lot of transformer-based models appeared to be great
	# at this task. One of the most well know is the GPT-2 model which was
	# trained on massive unsupervised text, that generates quite
	# impressive text. Another major breakthrough appeared when OpenAI
	# released the GPT-3 paper and its capabilities. This model is too
	# massive and is more than 1,400 times larger than its previous
	# version (GPT-2). Unfortunately, we cannot use GPT-3 as OpenAI did not
	# release the model weights, and even if it did, we as normal people
	# won't be able to have a machine that can load the model weights into
	# memory because it's too large.
	# Luckily, EleutherAI did a great job at trying to mimic the
	# capabilities of GPT-3 by releasing the GPT-J model. GPT-J has 6
	# billion parameters consisting of 28 layers with a dimension of 4096.
	# It was pre-trained on the Pile dataset, which is a large-scale
	# dataset created by EleutherAI itself. The Pile dataset is massive
	# with a size of over 825GB, consisting of 22 sub-datasets which
	# include Wikipedia English (6.36GB), GitHub (95.16GB), Stack Exchange
	# (32.2GB), ArXiv (56.21GB), and more. This explains the amazing
	# performance of GPT-J that you'll hopefully discover in this
	# tutorial.
	# The below table shows some useful models along with their number of
	# parameters and size. Choose the largest one that can fit into your
	# environment's memory:
	# 	Model			# Parameters		Size
	#	gpt2			124M			523MB
	# EleutherAI/gpt-neo-125M	125M			502MB
	# EleutherAI/gpt-neo-1.3B	1.3B			4.95GB
	# EleutherAI/gpt-neo-2.7B	2.7B			9.94GB
	# EleutherAI/gpt-j-6B		6B			22.5GB
	# The EleutherAI/gpt-j-6B model has 22.5GB of size, so make sure that
	# have atleast a memory of more than that amount to be able to perform
	# inference on this model. The good news is that Google Colab with the
	# High-RAM option works. If you are not able to load that big model,
	# you can try the other smaller versions such as
	# EleutherAI/gpt-neo-2.7B or EleutherAI/gpt-neo-1.3B.
	# Note that this is different from generating AI chatbot conversations
	# using models such as DialoGPT.

	# Using the standard GPT-2 model
	# Download & load the GPT-2 model.
	gpt2_generator = pipeline("text-generation", model="gpt2")

	# Use GPT-2 to generate 3 different sentences by sampling the top 50
	# candidates.
	sentences = gpt2_generator(
		"To be honest, neural networks",
		do_sample=True,
		top_k=50,
		temperature=0.6,
		max_length=128,
		num_return_sequences=3,
	)
	for sentence in sentences:
		print(sentence["generated_text"])
		print("=" * 50)

	# Setting top_k to 50 means we pick the 50 highest probability
	# vocabulary tokens to keep for filtering. We also decreased
	# temperature to 0.6 (default is 1.0) to increase the probability of
	# picking higer probability tokens. Setting it to 0 is the same as
	# greedy search (ie picking the most probable token). Notice that
	# some trailing sentences can be cut and not completed. You can always
	# increase the max_length to generate more tokens.
	# By passing the input text to the TextGenerationPipeline (pipeline
	# object), we're passing the arguments to the model.generate() method.
	# Therefore, it is highly suggested that you check the parameters of
	# the model.generate() method reference for a more customized
	# generation. Also read this blog post
	# (https://huggingface.co/blog/how-to-generate) explaining most of
	# the decoding techniques the method offers.

	# Using GPT-J/Neo
	gpt_neo_generator = pipeline(
		"text-generation", model="EleutherAI/gpt-neo-1.3B"
	)

	# Generate sentences with top-k sampling.
	sentences = gpt_neo_generator(
		"To be honest, robots will", 
		do_sample=True,
		top_k=50,
		temperature=0.6,
		max_length=128,
		num_return_sequence=3
	)
	for sentence in sentences:
		print(sentence["generated_text"])
		print("=" * 50)

	# Since GPT-J and other EleutherAI pre-trained models are trained on
	# the Pile dataset, it can noy only generate English text, but it can
	# talk anything. Let's try to generate python code:
	print(gpt_neo_generator('''
import os
# make a list of all african countries
''',
		do_sample=True,
		top_k=10,
		temperature=0.05,
		max_length=256,
	)[0]["generated_text"])

	# Notice that the temperature is lowered to 0.05, as this is not
	# really an open-ended generation. We want the African countries to be
	# correct as well as the Python syntax.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
