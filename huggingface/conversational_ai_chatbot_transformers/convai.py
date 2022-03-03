# convai.py
# Use Huggingface transformers library to generate conversational
# responses with pre-trained DialoGPT model in Python.
# Source: https://www.thepythoncode.com/article/conversational-ai-
# chatbot-with-huggingface-transformers-in-python 
# Windows/MacOS/Linux
# Python 3.7


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
	# Chatbots have gained a lot of popularity in recent years. As the
	# interest grows in using chatbots for business, researchers also did
	# a great job on advancing conversational AI chatbots.
	# This tutorial will use the huggingface transformers library to
	# employ the pre-trained DialoGPT model for conversational response
	# generation.
	# DialoGPT is a large-scale tunable neural conversational response
	# generation model trained on 147M conversations extracted from
	# Reddit. The good thing is that you can fine-tune it with your
	# dataset to achieve better performance than training from scratch.
	# This tutorial is about text generation in chatbots and not regular
	# texts. If you want open-ended generation, see this tutorial
	# (https://www.thepythoncode.com/article/text-generation-with-transformers-in-python)
	# which uses GPT-2 and GPT-J models to generate impressive text.

	#model_name = "microsoft/DialoGPT-large"
	model_name = "microsoft/DialoGPT-medium"
	#model_name = "microsoft/DialoGPT-small"
	tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cached_model")
	model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cached_tokenizer")
	
	# There are three versions of DialoGPT: small, medium, and large. Of
	# course, the larger, the better, but if you run this on your machine,
	# the small or medium will fit your memory with no problems. Loading
	# the model takes about 5GB of RAM. You can use Google Colab to try
	# out the large one.

	# Generating responses with Greedy Search
	# Use the greed search algorithm
	# (https://en.wikipedia.org/wiki/Greedy_algorithm) to generate
	# responses. We select the chatbot response with the highest
	# probability of choosing on each time step.

	# Chatting 5 times with greedy search.
	print("Greedy search")
	for step in range(5):
		# Take user input.
		text = input(">> You:")
		
		# Encode the input and add end of string token.
		input_ids = tokenizer.encode(
			text + tokenizer.eos_token, return_tensors="pt"
		)

		# Concatenate new user input with chat history (if there is).
		bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

		# Generate a bot response.
		chat_history_ids = model.generate(
			bot_input_ids,
			max_length=1000,
			pad_token_ids=tokenizer.eos_token_id,
		)
	
		# Print the output.
		output = tokenizer.decode(
			chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
			skip_special_tokens=True
		)
		print(f"DialoGPT: {output}")
	print("=" * 50)

	# An explanation of the above code:
	# -> First take the input from the user for chatting.
	# -> Encode the text to input_ids using the DialoGPT tokenizer, and 
	#	also append the end of the string token and return it as a pytorch
	#	tensor.
	# -> If this is the first time chatting with the bot, directly feed 
	#	input_ids to the model for generation. Otherwise, append the chat
	#	history using concatenation with the help of the torch.cat() method.
	# -> Use the model.generate() method for generating the chatbot response.
	# -> Lastly, as the returned output is a tokenized sequence too,
	#	decode the sequence using the tokenizer.decode() and set
	#	skip_special_tokens to True to make sure we don't see any annoying
	#	special tokens such as <|endoftext|>. Also, since the mdoel returns
	#	the whole sequence, we skip the previous chat history and print only
	#	the newly generated chatbot answer.
	# you see the model repeats a lot of responses, as these are the
	# highest probability and it is choosing it every time. By default,
	# model.generate() uses greedy search algorithm when no other
	# parameters are set.

	# Generating response with beam search
	# Beam search (https://en.wikipedia.org/wiki/Beam_search) allows us to
	# reduce the risk of missing highly probable sequences by keeping the
	# most likely num_beams of hypotheses at each time step and then
	# taking the sequences that have the overall highest probability, 
	# below code will generate chatbot responses with beam search.

	# Chatting 5 times with beam search.
	print("Beam search")
	for step in range(5):
		# Take user input.
		text = input(">> You:")
		
		# Encode the input and add end of string token.
		input_ids = tokenizer.encode(
			text + tokenizer.eos_token, return_tensors="pt"
		)

		# Concatenate new user input with chat history (if there is).
		bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

		# Generate a bot response.
		chat_history_ids = model.generate(
			bot_input_ids,
			max_length=1000,
			num_beams=3,
			early_stopping=True,
			pad_token_ids=tokenizer.eos_token_id,
		)
	
		# Print the output.
		output = tokenizer.decode(
			chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
			skip_special_tokens=True
		)
		print(f"DialoGPT: {output}")
	print("=" * 50)

	# When setting num_beams to 3 in model.generate() method, then we're
	# going to select three words at each time step and develop them to
	# find the highest overall probability of the sequence, setting
	# num_beams to 1 is the same as greedy search.

	# Generating responses with sampling
	# Beam and greedy search are great for tasks such as machine
	# translation or text summarization where the output is predictable.
	# However, it is not the best option for an open-ended generation as
	# in chatbots. For better generation, we need to introduce some
	# randomness where we sample from a wide range of candidate sequences
	# based on probabilities.

	# Chatting 5 times with sampling.
	print("Sampling")
	for step in range(5):
		# Take user input.
		text = input(">> You:")
		
		# Encode the input and add end of string token.
		input_ids = tokenizer.encode(
			text + tokenizer.eos_token, return_tensors="pt"
		)

		# Concatenate new user input with chat history (if there is).
		bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

		# Generate a bot response.
		chat_history_ids = model.generate(
			bot_input_ids,
			max_length=1000,
			do_sample=True,
			top_k=0,
			pad_token_ids=tokenizer.eos_token_id,
		)
	
		# Print the output.
		output = tokenizer.decode(
			chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
			skip_special_tokens=True
		)
		print(f"DialoGPT: {output}")
	print("=" * 50)

	# This time, we set do_sample to True for sampling, and top_k to 0
	# indicating we're selecting all possible probabilities.
	# There are some improvements. However, sampling on an exhaustive list
	# of sequences with low probabilities can lead to random generation.
	# To improve further, we can:
	# -> Lower the sampling temperature that helps us decrease the
	#	likelihood of picking low probability words and increase the
	#	likelihood of picking high probability words.
	# -> Use top-k sampling instead of picking all probable occurances.
	#	This will help us discard low probability words from getting picked.

	# Chatting 5 times with Top K sampling & tweaking temperature.
	print("Top k sampling")
	for step in range(5):
		# Take user input.
		text = input(">> You:")
		
		# Encode the input and add end of string token.
		input_ids = tokenizer.encode(
			text + tokenizer.eos_token, return_tensors="pt"
		)

		# Concatenate new user input with chat history (if there is).
		bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

		# Generate a bot response.
		chat_history_ids = model.generate(
			bot_input_ids,
			max_length=1000,
			do_sample=True,
			top_k=100,
			temperature=0.75,
			pad_token_ids=tokenizer.eos_token_id,
		)
	
		# Print the output.
		output = tokenizer.decode(
			chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
			skip_special_tokens=True
		)
		print(f"DialoGPT: {output}")
	print("=" * 50)

	# Now, we set top_k to 100 to sample from the top 100 words sorted
	# descending by probability. We also set temperature to 0.75 (default
	# is 1.0) to give a higher chance of picking high probability words,
	# setting the temperature to 0.0 is the same as greedy search; setting
	# it to infinity is the same as completely random. As you can see, it
	# is much better now; feel free to tweak temperature and top_k
	# parameters and see if it can improve.

	# Nucleus sampling
	# Nucleus sampling or Top-p sampling chooses from the smallest
	# possible words whose cumulative probability exceeds the parameter p
	# we set.

	# Chatting 5 times with nucleus sampling & tweaking temperature.
	print("Nucleus sampling")
	for step in range(5):
		# Take user input.
		text = input(">> You:")
		
		# Encode the input and add end of string token.
		input_ids = tokenizer.encode(
			text + tokenizer.eos_token, return_tensors="pt"
		)

		# Concatenate new user input with chat history (if there is).
		bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

		# Generate a bot response.
		chat_history_ids = model.generate(
			bot_input_ids,
			max_length=1000,
			do_sample=True,
			top_p=0.95,
			top_k=0,
			temperature=0.75,
			pad_token_ids=tokenizer.eos_token_id,
		)
	
		# Print the output.
		output = tokenizer.decode(
			chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
			skip_special_tokens=True
		)
		print(f"DialoGPT: {output}")
	print("=" * 50)

	# We set top_k to 0 to disable Top-k sampling, but you can use both
	# methods, which works better. Now the chatbot clearly makes sense in
	# a lot of cases.

	# Now let's add some code to generate more than one chatbot response,
	# and then we choose which response to include in the next input:

	# Chatting 5 times with nucleus sampling & top-k sampling & tweaking
	# temperature multiple sentences.
	print("nucleas + top-k sampling")
	for step in range(5):
		# Take user input.
		text = input(">> You:")
		
		# Encode the input and add end of string token.
		input_ids = tokenizer.encode(
			text + tokenizer.eos_token, return_tensors="pt"
		)

		# Concatenate new user input with chat history (if there is).
		bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

		# Generate a bot response.
		chat_history_ids_list = model.generate(
			bot_input_ids,
			max_length=1000,
			do_sample=True,
			top_p=0.95,
			top_k=50,
			temperature=0.75,
			num_return_sequences=5,
			pad_token_ids=tokenizer.eos_token_id,
		)
	
		# Print the output.
		for i in range(len(chat_history_ids_list)):
			output = tokenizer.decode(
				chat_history_ids_list[i][bot_input_ids.shape[-1]:], 
				skip_special_tokens=True
			)
			print(f"DialoGPT {i}: {output}")
		choice_index = int(input("Choose the response you want for the next input: "))
		chat_history_ids = torch.unsqueeze(chat_history_ids_list[choice_index], dim=0)
	print("=" * 50)

	# Setting the num_return_sequences to 5 will return five sentences at
	# a time. We have to choose the one included in the following sequence.

	# Conclusion
	# For more information on generating text, read the How to generate
	# text with transformers (https://huggingface.co/blog/how-to-generate)
	# guide. Also a greate and exciting challange for you is combining
	# this with text-to-speech and speech-to-text tutorials to build a
	# virtual assistant like Alexa, Siri, and Cortana.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
