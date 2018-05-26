import json, codecs, os

from config import Config

# Load a set of words, such as all English words or all DSTs.
def load_wordset(filename):
	with open(filename, "r") as f:
		lines = f.read().splitlines() 
	return set(lines)

# Load a dataset, such as train_data.json.
def load_dataset(filename):
	with codecs.open(filename, 'r', 'utf-8') as f:
		return json.loads(f.read())


# Create the relevant folders if they do not already exist.
def create_folders(folders):
	def make_folder(folder):
		if not os.path.isdir(folder):
			print "Creating folder '%s'..." % folder
			os.makedirs(folder)		
	for folder in folders:
		make_folder(folder)	


config = Config()

word_set = load_wordset(config.words_en)


def get_word_class(word, label):
	class_of_word = "O"
	if word not in word_set:# and word.isalpha():
		if word == label:
			class_of_word = "DST"
		else:
			if " " in label > 0 and word == ''.join([l[0] for l in label.split(" ")]):
				class_of_word = "ACR"
			else:
				class_of_word = "SPE"
	return class_of_word