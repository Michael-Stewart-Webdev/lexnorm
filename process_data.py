# -*- coding: utf-8 -*-

# Michael Stewart
# 29 April 2018

from colorama import Fore, Style
from global_vars import *
import random
import codecs, json, os
from collections import defaultdict
from config import Config
from utils import *
import gensim, re
from collections import Counter
from itertools import groupby, chain
import sys

from libraries.sequence_tagging import build_data as seqtag_build
from libraries.sequence_tagging import config as seqtag_config
from libraries.sequence_tagging import main as seqtag_main

from shutil import copyfile

config = None
owd = os.getcwd()


class Preprocessor:	   

	def __init__(self, conf=None, reset_all=False, retrain_bilstm=False, retrain_word2vec=False):

		self.RESET_ALL = reset_all

		global config
		config = conf if conf else Config()

		# 0. Create the necessary folders they do not already exist.
		self.create_asset_and_data_folders()

		# 1. Load the datasets.
		self.word_set = load_wordset(config.words_en)		
		train_data = load_dataset(config.train_data)
		test_data  = load_dataset(config.test_data)
		truth_data = load_dataset(config.truth_data)
		test_truth_data = self.combine_test_truth(test_data, truth_data)

		# 2. Determine the DSTs in the training dataset and write them to the dsts file.
		#self.dsts = self.determine_dsts(train_data)

		# 3. Create the class datasets to be used by the Bi-LSTM-CRF and write them to the class data files.
		self.create_class_datasets(train_data, test_truth_data)

		# 4. Pad the embedding data with SOS and EOS tokens.
		# TODO: Remove this, probably.
		# self.pad_embedding_data()

		# 5. Create the unique replacements file (for dictionary normalisation).
		unique_replacements = self.create_unique_replacements(train_data)

		# 6. Build the personal word file for Aspell to use.
		self.build_pwl_file_for_aspell(train_data)

		# 7. Set up the classifier if necessary.
		self.set_up_classifier()

		# 8. Run the classifier. The classifier's predictions will be saved to a file.
		self.run_classifier(test_data)

		self.save_train_data_for_word2vec(train_data)

		# 9. Train the word2vec model on the embedding data file.
		self.train_word2vec()

		# 10. Build the dictionary mapping each acronym to its most likely replacement.
		self.build_acronym_possibilities(train_data, test_data, unique_replacements)


		# TODO: Move all QRNN-related functions into a separate program, such as process_data_qrnn.py or something.

		# 11. Build the training dataset from the QRNN using the training dataset + an optional supplemental dataset.
		self.build_qrnn_dataset(train_data)

		# 12. Segment the test set into smaller sentences for the QRNN.
		self.segment_testset_for_qrnn(test_data)

		# 13. Run the QRNN ......
		# TODO: Run the QRNN from here.


	def create_asset_and_data_folders(self):
		create_folders(
			[ 	config.asset_path + config.experiment_name,
				config.class_data_folder,
				config.lstm_classifier_predictions_folder,
				config.classifier_predictions_folder,
				config.qrnn_predictions_folder,
				config.unique_replacements_folder,
				config.acronyms_folder,
				config.emb_model_folder,
				config.aspell_pwl_folder,
				config.embedding_data_train_folder,
				config.qrnn_training_dataset_folder,
			])
		

	# Combine the test_data and truth_data into one dataset.
	def combine_test_truth(self, test_data, truth_data):
		if len(test_data) != len(truth_data):
			raise ValueError("Test dataset and truth dataset must be of the same length.")
 		combined_data = []
 		for x in range(len(test_data)):
 			combined_data.append( { "index": test_data[x]["index"], "input": test_data[x]["input"], "output": [y for y in truth_data[x]["output"]] } )
 		return combined_data

 	# Pad the data to be used for embeddings with _SOS_ and _EOS_ characters.
	# def pad_embedding_data(self):
	# 	if not os.path.exists(config.padded_emb_filename) or self.RESET_ALL:
	# 		print "Padding embedding data..."
	# 		sents = []
	# 		with codecs.open(config.embeddings_data, 'r', 'utf-8') as f:
	# 			sents = [config.SOS_TOKEN + line.rstrip() + config.EOS_TOKEN for line in f]
	# 		with codecs.open(config.padded_emb_filename, 'w', 'utf-8') as f:
	# 			f.write("\n".join(sents))

	# Determine all the DSTs in the dataset.
	# A DST is a token where the input token == the output token.
	# Writes the DSTS to the config.dsts_filename file when done.
	# def determine_dsts(self, train_data):	

	# 	if os.path.exists(config.dsts_filename) and not self.RESET_ALL:
	# 		return load_wordset(config.dsts_filename)
	# 	print "Building set of all Domain-Specific Terms..."
	# 	dsts = set()
	# 	for x in range(len(train_data)):
	# 		inp = train_data[x]["input"]
	# 		outp = train_data[x]["output"]
	# 		if len(inp) != len(outp):
	# 			raise ValueError("The length of the corresponding input and output sentences in the training data must be the same.")
	# 		for i in range(len(inp)):
	# 			if inp[i] == outp[i] and inp[i] not in self.word_set and inp[i].isalpha():
	# 				dsts.add(inp[i])

	# 	with codecs.open(config.dsts_filename, 'w', 'utf-8') as f:
	# 		f.write("\n".join([dst for dst in dsts]))
	# 	return dsts

	# Creates data for the acronym/spelling error/dst classifier
	def create_class_datasets(self, train_data, test_truth_data):
		print "Building class datasets..."

		def build_dataset(dataset):
			class_dataset = []
			for doc in dataset:
				class_dataset.append([])
				for x in range(len(doc["input"])):
					word =  doc["input"][x]
					label = doc["output"][x]
					class_of_word = get_word_class(word, label)
					class_dataset[-1].append((word, class_of_word))
			return class_dataset

		def write_dataset(dataset, filename):
			with codecs.open("%s/%s.txt" % (config.class_data_folder, filename), 'w', 'utf-8') as f:
				for sentence in dataset:
					for wordpair in sentence:
						f.write("%s %s\n" % (wordpair[0], wordpair[1]))
					f.write("\n")
				f.write("\n")

		train_classes_out = build_dataset(train_data)
		test_classes_out  = build_dataset(test_truth_data)

		# Split train set into dev set
		dev_classes_out   = train_classes_out[int(len(train_classes_out) * 0.9) : ]
		train_classes_out = train_classes_out[ : int(len(train_classes_out) * 0.9)]

		write_dataset(train_classes_out, "train")
		write_dataset(dev_classes_out, "dev")
		write_dataset(test_classes_out, "test")

		# Build a symbolic link in the sequence tagger's dataset folder to avoid having to copy all the datasets into that directory.
		if os.path.exists("libraries/sequence_tagging/data/datasets"):
			os.remove("libraries/sequence_tagging/data/datasets")
		os.symlink("../../../" + config.class_data_folder, "libraries/sequence_tagging/data/datasets")

	# Finds all special tokens (domain-specific terms, direct replacements, and acronyms) in the data.
	def create_unique_replacements(self, train_data):
		if os.path.exists(config.unique_replacements_filename) and not self.RESET_ALL:
			return load_dataset(config.unique_replacements_filename)

		print "Building unique replacements for dictionary normalisation..."
		replacements = defaultdict(set)

		for document in train_data:
			di = document["input"]
			do = document["output"]	
			for x in range(len(di)):
				if di[x] not in self.word_set:
					#if di[x] != do[x] and di[x].isalpha():			
					replacements[di[x]].add(do[x].lower())

		# Remove all ambiguous replacements
		# If something maps to more than one token, it shouldn't be used for normalisation.
		unique_replacements = {}
		for r in replacements:
			if len(replacements[r]) == 1:
				unique_replacements[r] = list(replacements[r])[0]

		# Write the unique replacements to the file
		with codecs.open(config.unique_replacements_filename, 'w', 'utf-8') as f:
			json.dump(unique_replacements, f)

		return unique_replacements

	# Build the personal word list file for Aspell to use.
	# This list will contain the unique correct words in the training data that are not present in the word list.
	# (most of them should be domain-specific terms etc).
	def build_pwl_file_for_aspell(self, train_data):
		print "Building personal word list for Aspell..."
		pwl = set()
		for document in train_data:
			do = document["output"]
			for x in range(len(do)):
				if do[x] not in self.word_set:
					pwl.add(do[x])
		with open(config.aspell_pwl_file, 'w') as f:
			f.write("\n".join([w for w in pwl]))

	# Set up the classifier, such as building the vocab for the bi-lstm-crf.
	def set_up_classifier(self):
		#if self.RESET_ALL:
		if (not os.path.exists(config.lstm_classifier_predictions_filename)) or self.RESET_ALL:
			print "Setting up classifier..."
			os.chdir("libraries/sequence_tagging")
			seqtag_build.build_data(seqtag_config.Config())
			os.chdir(owd)
		else:
			print "No need to set up classifier, '%s' already exists." % config.classifier_predictions_filename
		
	# Run the classifier.
	def run_classifier(self, test_data):
		if (not os.path.exists(config.none_classifier_predictions_filename)) or self.RESET_ALL:
			print "Running 'everything is a spelling error' classifier..."
			outp = []
			for d in test_data:
				outp.append([(w, "SPE") for w in d["input"]])
			with open(config.none_classifier_predictions_filename, 'w') as f:
				for d in outp:
					for x in d:
						f.write("%s %s\n" % (x[0], x[1]))
					f.write("\n")
		if (not os.path.exists(config.lstm_classifier_predictions_filename)) or self.RESET_ALL:
			#if config.classification_strategy == BI_LSTM_CRF:
				print "Running Bi-LSTM-CRF classifier..."
				os.chdir("libraries/sequence_tagging")
				seqtag_main.main(seqtag_config.Config())
				os.chdir(owd)				
				copyfile(config.lstm_classifier_predictions_unmoved_filename, config.lstm_classifier_predictions_filename)
		else:
			"No need to run classifier, '%s' already exists." % config.classifier_predictions_filename


 	# Save the train data to the embeddings train file.
 	def save_train_data_for_word2vec(self, train_data):
 		#if os.path.exists(config.embedding_data_train):
		#	print "No need to save training data for embeddings, %s already exists." % config.embedding_data_train
		#	return
		if config.embedding_model_pretrained:
			return

		if not os.path.exists(config.embedding_data_train_all):
			copyfile(config.embeddings_data_file, config.embedding_data_train_all)

 		if os.path.exists(config.embedding_data_train):
			print "No need to save training data for embeddings, %s already exists." % config.embedding_data_train
			return
		txt_docs = []
		for doc in train_data:
			groups = [list(group) for k, group in groupby(doc["output"], lambda x: x == ".") if not k]
			for g in groups:
				txt_docs.append(" ".join(g))

		with codecs.open(config.embedding_data_train, 'w', 'utf-8') as f:
			f.write("\n".join(txt_docs))

		
	# Train the embedding model via Gensim's Word2Vec.
	def train_word2vec(self):

		if config.embedding_model_pretrained:
			print "Using pretrained embedding model."
			return

		class MySentences(object):
		    def __init__(self, dirname):
		        self.dirname = dirname
		 
		    def __iter__(self):
		        for fname in os.listdir(self.dirname):
		            for line in open(os.path.join(self.dirname, fname)):
		            	l = line.split()
		                yield [w.lower() for w in line.split() if w.isalpha() and w not in stopwords]
		stopwords = load_wordset(config.stopwords_file)
		sentences = MySentences(config.embedding_data_train_folder)


		if(os.path.exists(config.emb_model_filename)) and not self.RESET_ALL:
			print "No need to run word2vec, '%s' already exists." % config.emb_model_filename
			model = gensim.models.Word2Vec.load(config.emb_model_filename)
		else:
			print "Training word2vec model..."
			model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=1, max_vocab_size=None, sg=0)
			#model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=1, max_vocab_size=None, sg=1)
			model.save(config.emb_model_filename)

	# Builds all acronym possibilities for the dataset.
	def build_acronym_possibilities(self, train_data, test_data, unique_replacements):
		print "Building potential acronyms..."
		acronym_data = {}

		potential_acronyms = set()
		for x in range(len(test_data)):
			inp = test_data[x]["input"]
			for w in inp:
				if w not in self.word_set and w not in unique_replacements:
					potential_acronyms.add(w)

		token_list = []
		for document in train_data:
			di = document["output"]
			for word in di:
				token_list.append(word.lower())
		all_tokens = " ".join(token_list)

		for token in potential_acronyms:
			regex_string = ' '.join([r'\b%s\w+' % re.escape(c) for c in token])			
			matches = re.findall(r'%s' % regex_string, all_tokens)
			if len(matches) > 0:
				mc = Counter(matches).most_common()[0][0]
			else:
				mc = token
			acronym_data[token] = mc

		with codecs.open(config.acronyms_filename, 'w', 'utf-8') as f:
			json.dump(acronym_data, f)



	# Build the dataset for the QRNN to train from.
	# If config.qrnn_dataset_supplemental is provided, the supplemental dataset will be 'corrupted' and used
	# as part of the training data.
	def build_qrnn_dataset(self, train_data):
		if os.path.isfile(config.qrnn_training_dataset_filename_en) and os.path.isfile(config.qrnn_training_dataset_filename_co):
			print "No need to build QRNN dataset, as it is already present."
			return

		print "Building dataset for QRNN..."
		train_out_co = []
		train_out_en = []



		# Builds the 'supplemental dataset'. The supplemental dataset should be a large unlabeled corpus of similar data.
		def build_supplemental_dataset():
			supplemental_dataset_co = []
			supplemental_dataset_en = []
			errors_dict = []

			# 'Corrupts' a word, i.e. mimicks spelling errors.
			# Occasionally joins the word with the previous or next word, or splits the word into two words.
			def corrupt_word(word):

				if len(word) < 2 or not word.isalpha():
					return word

				def corrupt_again(word):
					if random.random() <= config.qrnn_supplemental_corruption_coefficient_2:
						return corrupt_word(word)
					else:
						return word

				choices = [1, 2, 3, 4]
				choice = random.choice(choices)

				# 1. Check for common misspellings of the word, and replace the word with a common misspelling.
				# If it is not present in the dictionary, try corrupt it again.
				if choice == 1:			
					return random.choice(errors_dict[word]) if word in errors_dict else corrupt_again(word)
				# 2. Duplicate a letter somewhere
				elif choice == 2:
					ind = random.randint(0, len(word)-1)
					word = word[:ind] + word[ind] + word[ind:]
					return corrupt_again(word)
				# 3. Remove a letter somewhere
				elif choice == 3:
					ind = random.randint(0, len(word)-1)
					word = word[:ind] + word[ind+1:]
					return corrupt_again(word)
				# 4. Swap two letters
				elif choice == 4:
					if len(word) == 1:
						return word
					if len(word) > 2:
						ind = random.randint(0, len(word) - 2)
					else:
						ind = 0
					word = word[:ind] + word[ind+1] + word[ind] + word[ind+2:]
					return corrupt_again(word)

			# Corrupts a sentence by joining two words together (replacing space with underscore in the test sentence).
			# The probability of this occuring is governed by config.qrnn_supplemental_corruption_coefficient_3.
			def corrupt_sentence(sent_co, sent_en):
				space_indices_co = [pos for pos, c in enumerate(sent_co) if c == " "]
				space_indices_en = [pos for pos, c in enumerate(sent_en) if c == " "]
				chars_removed = 0
				for i, pos in enumerate(space_indices_co):
					if random.random() < config.qrnn_supplemental_corruption_coefficient_3:
						sent_co = sent_co[:pos-chars_removed] + sent_co[pos+1-chars_removed:]
						s = list(sent_en)
						s[space_indices_en[i]] = "_"
						sent_en = "".join(s)
						chars_removed += 1
				return sent_co, sent_en

			with codecs.open(config.qrnn_supplemental_dataset, 'r', 'utf-8') as f:
				supplemental_dataset = [line.strip().split(" ") for line in f if len(line) < 150]

			# Load the dictionary of common errors, mapping a word to its common erroreous spelling.	
			errors_dict = defaultdict(list)
			with open(config.qrnn_supplemental_common_errors_dict, 'r') as f:
				lines = f.read().splitlines()
				for line in lines:
					line = line.split("->")
					errors_dict[line[1]].append(line[0])

			print "Corrupting the supplemental dataset... (%d sentences total)" % len(supplemental_dataset)

			for i, sent in enumerate(supplemental_dataset):
				corrupted_sent = " ".join([ corrupt_word(w) if random.random() <= config.qrnn_supplemental_corruption_coefficient_1 else w for w in sent ])
				sent_co, sent_en = corrupt_sentence(corrupted_sent, " ".join(sent))
				supplemental_dataset_co.append(sent_co)
				supplemental_dataset_en.append(sent_en)
				if i % (len(supplemental_dataset) / 100) == 0:
					print i,
					sys.stdout.flush()

			return supplemental_dataset_co, supplemental_dataset_en

		if config.qrnn_supplemental_dataset:
			supplemental_dataset_co, supplemental_dataset_en = build_supplemental_dataset()
		else:
			supplemental_dataset_co, supplemental_dataset_en = []

		# for doc in train_data:
		# 	inp = " ".join(doc["input"]).split(" . ")
		# 	for i in inp:
		# 		train_out_co.append(i.replace(" .", ""))
		# 	inp = " ".join([w.replace(" ", "_") for w in doc["output"]]).split(" . ")
		# 	for i in inp:
		# 		train_out_en.append(i.replace(" .", ""))
		for sent in supplemental_dataset_co:
			train_out_co.append(sent)
		for sent in supplemental_dataset_en:
			train_out_en.append(sent)

		if len(train_out_co) != len(train_out_en):
			raise Exception("The training and testing output files do not appear to be the same length.")
 
		with codecs.open(config.qrnn_training_dataset_filename_en, 'w', 'utf-8') as f:
			f.write("\n".join(train_out_en))
		with codecs.open(config.qrnn_training_dataset_filename_co, 'w', 'utf-8') as f:
			f.write("\n".join(train_out_co))



	# Segment the data for the QRNN, and save a file containing information about how it was segmented.
	# For example, the document "Hello there. My name, of course, is Michael", would be segmented into
	# Hello there
	# My name
	# of course
	# is Michael
	# ... and the segmentation file would contain [".", ".", ",", "N"]
	def segment_testset_for_qrnn(self, test_data):
		print "Segmenting the test set for the QRNN..."
		segmented_data     = []
		segmented_data_all = [] # contains all data (including overly long sents)
		segmented_metadata = []
		c = 0
		for x in range(len(test_data)):

			c += 1
			inp = test_data[x]["input"]


			for i, x in enumerate(inp):
				if (x == "." or x == ",") and i < len(inp)-1:
					segmented_metadata.append(x)
			groups = [list(group) for k, group in groupby(inp, lambda x: x == "." or x == ",") if not k]
			for g in groups:


				if len(' '.join(g)) > 157:
					segmented_data.append(config.qrnn_sent_too_long_char)
				else:
					segmented_data.append(g)
				segmented_data_all.append(g)

			if inp[-1] == ".":
				segmented_metadata.append(".N")
			elif inp[-1] == ",":
				segmented_metadata.append(",N")
			else:
				segmented_metadata.append("N")

			debug = False 
			if debug:
				print "---" 
				print "INPUT:"
				print " ".join([Fore.YELLOW + "[" + x + "]" + Style.RESET_ALL if x in [",", "."] else x for x in inp])
				print "---" 

				print "GROUPS:"
				for g in groups:
					print g 
				print "---" 
				print "SEGMENTED_DATA:"
				print segmented_data[-1]
				print "---" 

				print "SEGMENTED_METADATA"
				print segmented_metadata
				raw_input()

		with codecs.open(config.qrnn_test_input_file_all, "w", 'utf-8') as f:
			f.write("\n".join([(" ").join(s) for s in segmented_data_all]))
		with codecs.open(config.qrnn_test_input_file, "w", 'utf-8') as f:
			f.write("\n".join([(" ").join(s) for s in segmented_data]))
		with open(config.qrnn_segmentation_metadata_file, "w") as f:
			json.dump(segmented_metadata, f)





if __name__ == "__main__":
	preprocess_data = Preprocessor()