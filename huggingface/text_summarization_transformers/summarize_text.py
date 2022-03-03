# summarize_text.py
# Use huggingface transformers and pytorch libraries to summarize long
# text, using the pipeline API and T5 transformer model in Python.
# Source: https://www.thepythoncode.com/article/text-summarization-
# using-huggingface-transformers-python
# Windows/MacOS/Linux
# Python 3.7


import torch
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer


def main():
	# Text summarization is the task of shortening long pieces of text
	# into a concise summary that preserves key information content and
	# overall meaning. There are two different approaches that are widely
	# used for text summarization:
	# -> Extractive Summarization: This is where the model identifies the
	#	meaningful sentences and phrases from the original text and only
	#	outputs those.
	# -> Abstractive Summarization: The model produces an entirely 
	#	different text shorter than the original. It generates new sentences
	#	in a new form, just like humans do.
	# This example will use huggingface's transformers library in Python
	# to perform abstractive text summarization on any text we want. We
	# chose huggingface transformers because it provides us with thousands
	# of pre-trained models not just for text summarization but for a wide
	# variety of NLP tasks such as text classification, text paraphrasing,
	# question answering, machine translation, text generation, chatbots,
	# and more.

	# Using the pipeline API
	# The most straight forward way to use models in transformers is using
	# the pipeline API.

	# Using pipeline API for summarization task.
	summarization = pipeline("summarization")
	original_text = """
Paul Walker is hardly the first actor to die during a production. 
But Walker's death in November 2013 at the age of 40 after a car crash was especially eerie given his rise to fame in the "Fast and Furious" film franchise. 
The release of "Furious 7" on Friday offers the opportunity for fans to remember -- and possibly grieve again -- the man that so many have praised as one of the nicest guys in Hollywood. 
"He was a person of humility, integrity, and compassion," military veteran Kyle Upham said in an email to CNN. 
Walker secretly paid for the engagement ring Upham shopped for with his bride. 
"We didn't know him personally but this was apparent in the short time we spent with him. 
I know that we will never forget him and he will always be someone very special to us," said Upham. 
The actor was on break from filming "Furious 7" at the time of the fiery accident, which also claimed the life of the car's driver, Roger Rodas. 
Producers said early on that they would not kill off Walker's character, Brian O'Connor, a former cop turned road racer. Instead, the script was rewritten and special effects were used to finish scenes, with Walker's brothers, Cody and Caleb, serving as body doubles. 
There are scenes that will resonate with the audience -- including the ending, in which the filmmakers figured out a touching way to pay tribute to Walker while "retiring" his character. At the premiere Wednesday night in Hollywood, Walker's co-star and close friend Vin Diesel gave a tearful speech before the screening, saying "This movie is more than a movie." "You'll feel it when you see it," Diesel said. "There's something emotional that happens to you, where you walk out of this movie and you appreciate everyone you love because you just never know when the last day is you're gonna see them." There have been multiple tributes to Walker leading up to the release. Diesel revealed in an interview with the "Today" show that he had named his newborn daughter after Walker. 
Social media has also been paying homage to the late actor. A week after Walker's death, about 5,000 people attended an outdoor memorial to him in Los Angeles. Most had never met him. Marcus Coleman told CNN he spent almost $1,000 to truck in a banner from Bakersfield for people to sign at the memorial. "It's like losing a friend or a really close family member ... even though he is an actor and we never really met face to face," Coleman said. "Sitting there, bringing his movies into your house or watching on TV, it's like getting to know somebody. It really, really hurts." Walker's younger brother Cody told People magazine that he was initially nervous about how "Furious 7" would turn out, but he is happy with the film. "It's bittersweet, but I think Paul would be proud," he said. CNN's Paul Vercammen contributed to this report.
"""
	summary_text = summarization(original_text)[0]["summary_text"]
	print("Summary:", summary_text)

	# Note that the first time this is executed, with will download the
	# model architecture and the weights and tokenizer configuration. We
	# specify the "summarization" task to the pipeline and then simply
	# pass our long text to it.

	# Here is another example.
	print("=" * 50)
	original_text = """
For the first time in eight years, a TV legend returned to doing what he does best. 
Contestants told to "come on down!" on the April 1 edition of "The Price Is Right" encountered not host Drew Carey but another familiar face in charge of the proceedings. 
Instead, there was Bob Barker, who hosted the TV game show for 35 years before stepping down in 2007. 
Looking spry at 91, Barker handled the first price-guessing game of the show, the classic "Lucky Seven," before turning hosting duties over to Carey, who finished up. 
Despite being away from the show for most of the past eight years, Barker didn't seem to miss a beat.
"""
	summary_text = summarization(original_text)[0]["summary_text"]
	print("Summary:", summary_text)

	# As you can see, the model generated an entirely new summarized text
	# that does not belong to the orginal text. This is the quickest way
	# to use transformers.

	# Using T5 model

	# Initialize the model architecture and weights.
	model = T5ForConditionalGeneration.from_pretrained(
		"t5-base", cache_dir="./t5-for-summarization"
	)

	# Initialize the model tokenizer.
	tokenizer = T5Tokenizer.from_pretrained(
		"t5-base", cache_dir="./t5-for-summarization-tokenizer"
	)

	# The first time this is executed, the code will download the t5-base
	# model architecture, weights, tokenizer, vocabulary, and
	# configuration. Using the from_pretrained() method to load it as a 
	# pre-trained model, T5 comes with three versions in this library:
	# t5-small, which is a smaller version of t5-base, and t5-large, that
	# is larger and more accurate than the others. If you want to do
	# summarization in a different language other than English, and if
	# it's not available in the list of available models, consider
	# pre-training a model from scratch using your dataset.

	# Set the text we want to summarize.
	article = """
Justin Timberlake and Jessica Biel, welcome to parenthood. 
The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People. 
"Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports. 
The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both.
"""

	# Encode the text to be suitable for the model as an input.
	inputs = tokenizer.encode(
		"summarize: " + article,
		return_tensors="pt",
		max_length=512,
		truncation=True,
	)

	# We used the tokenizer.encode() method to convert the string text to
	# a list of integers, where each integer is a unique token. We also
	# set the max_length t0 512, indicating that we do not want the
	# original text to bypass 512 tokens. Likewise, we set return_tensors
	# to "pt" to get pytorch tensors as output. Notice that we prepended
	# the text with "summarize: ". This is because T5 isn't just for text
	# summarization. You can use it for any text-to-text transformation,
	# such as machine translation or question answering or even
	# paraphrasing.
	# For example, we can use the T5 transformer for machine translation,
	# and you can set "translate English to German: " instead of 
	# "summarize: " and you'll get German translation output (more
	# precisely, you'll get a summarized German translation, as you'll see
	# why in model.generate()).

	# Generate the summarization output.
	outputs = model.generate(
		inputs,
		max_length=150,
		min_length=40,
		length_penalty=2.0,
		num_beams=4,
		early_stopping=True,
	)

	# Just for debugging.
	print(outputs)
	print(tokenizer.decode(outputs[0]))

	# Going to the most exciting part, the parameters passed to the
	# model.generate() method are:
	# -> max_length: The maximum number of tokens to generate (we have
	#	specified a total of 150). You can change that if you want.
	# -> min_length: The minimum number of tokens to generate. Note that
	#	this will also work if you set it to another task, such as English
	#	to German translation.
	# -> length_penalty: Exponential penalty to the length. 1.0 means no
	#	penalty. Increasing this parameter will increase the size of the
	#	output text.
	# -> num_beams: Specifying this parameter will lead the model to use
	#	beam search instead of greedy search. Setting num_beams to 4 will
	#	allow the model to look ahead for four possible words (1 in the case
	#	of greedy search), to keep the most likely 4 hypotheses at each time
	#	step, and choosing the one that has the overall highest probability.
	# -> early_stopping: We set it to True so that generation is finished
	#	when all beam hypotheses reach the end of string token (EOS).
	# We then use the decode() method from the tokenizer to convert the
	# tensor back to human-readable text.

	# Conclusion
	# There are a lot of other parameters to tweak in the model.generate()
	# method. Check out this tutorial
	# (https://huggingface.co/blog/how-to-generate) from the huggingface
	# blog for more on generating outputs.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
