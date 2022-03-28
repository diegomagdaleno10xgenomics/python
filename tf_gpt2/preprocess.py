# preprocess.py
# Preprocess the arxiv-metadata-oai-snapshot.json file such that the
# contents can be read by python's JSON module.
# Source (Reference): https://www.freecodecamp.org/news/python-json-how-to-convert-a-string-to-json/
# Windows/MacOS/Linux
# Python 3.7


import os
import json


def main():
	# Make sure the target file exists.
	source = "arxiv-metadata-oai-snapshot.json"
	if not os.path.exists(source):
		print("Error: Could not find the required JSON file " + source + ".")

	# Open unformatted JSON file.
	with open(source, "r") as f:
		data = f.readlines()

	# Format the data into proper JSON.
	json_data = []
	for i in range(len(data)):
		json_data.append(json.loads(data[i].rstrip("\n")))

	# Write the formatted data to a new file.
	destination = "arxiv-metadata-snapshot.json"
	with open(destination, "w+") as f:
		json.dump(json_data, f)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
