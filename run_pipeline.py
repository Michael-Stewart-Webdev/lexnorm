from config import Config
from utils import *
import sys, os
from libraries.sequence_tagging import main as seqtag_main
from libraries.sequence_tagging import config as seqtag_config
from colorama import Fore, Style
from global_vars import *

#os.chdir("libraries/sequence_tagging")
#seqtag_main.main(seqtag_config.Config())
#seqtag_build.build_data(seqtag_config.Config())
#exit()
from nltk.corpus.reader import ConllCorpusReader

from libraries.spelling_error_normaliser import SpellingErrorNormaliser
from libraries.acronym_normaliser import AcronymNormaliser



config = None
owd = os.getcwd()

class LexnormPipeline():

	def __init__(self, conf=None, verbose=False, spellingErrorNormaliser=None, acronymNormaliser=None):
		global config
		config = conf if conf else Config()

		if not spellingErrorNormaliser:
			spellingErrorNormaliser = SpellingErrorNormaliser(config)		
		if not acronymNormaliser:
			acronymNormaliser = AcronymNormaliser(config)

		# 0. Create the output folders.
		create_folders(
			[	config.output_path,						
				config.output_path_exp,
				config.output_path_exp_full
			])

		# 1. Load the datasets.

		self.verbose = verbose

		self.word_set = load_wordset(config.words_en)		
		train_data = load_dataset(config.train_data)
		test_data  = load_dataset(config.test_data)
		truth_data = load_dataset(config.truth_data)

		prediction_data = self.init_prediction_data(test_data)

		# 2. Run dictionary normalisation if appropriate.

		#self.print_total_predictions(prediction_data)

		if config.use_dictionary_normalisation:
			unique_replacements = load_dataset(config.unique_replacements_filename)
			prediction_data = self.run_dictionary_normalisation(test_data, prediction_data, unique_replacements)
		else:
			unique_replacements = {}



		# 3. Use the classifiers' predictions to determine what actions to take for each token.
		prediction_data, normalised_tokens, normalised_documents = self.get_classifier_predictions(test_data, prediction_data, truth_data, unique_replacements, spellingErrorNormaliser, acronymNormaliser)

		#self.print_total_predictions(prediction_data)

		# 4. Save the predictions to a file.
		self.save_predictions_to_file(prediction_data, truth_data)
		self.save_normalised_tokens_to_file(normalised_tokens)
		self.save_normalised_documents_to_file(normalised_documents)


		

	# Initialises the prediction data array.
	# It contains None for every output token that is not an English word. These None tokens will be changed while the pipeline is running.
	def init_prediction_data(self, test_data):
		prediction_data = []
		for x in range(len(test_data)):				
			prediction_data.append( { "index": test_data[x]["index"], "input": [y for y in test_data[x]["input"]], "output": [y.lower() for y in test_data[x]["input"]] } )
 		return prediction_data

 	# Print the total number of predictions made by the pipeline.
 	def print_total_predictions(self, prediction_data):
 		c = 0
 		l = 0
 		for pred in prediction_data:
 			c += pred["output"].count(None)
 			l += len(pred["output"])
 		print "Found predictions for %d / %d tokens. (%.2f%%)" % (l-c, l, 100.0 * (l-c) / l)


	# Normalise dataset via Dictionary Normalisation.
	def run_dictionary_normalisation(self, test_data, prediction_data, unique_replacements):
		print "Running dictionary normalisation..."
		total = 0
		for x in range(len(test_data)):
			inp = test_data[x]["input"]
			for i in range(len(inp)):
				if inp[i] in unique_replacements:
					if inp[i].isalpha():
						total += 1					

					prediction_data[x]["output"][i] = unique_replacements[inp[i]]
		print "%d tokens replaced via dictionary." % total
		return prediction_data



	# Gets the predictions from the classifier
	def get_classifier_predictions(self, test_data, prediction_data, truth_data, unique_replacements, spellingErrorNormaliser, acronymNormaliser):
		# Build a ConllCorpusReader.
		corpus = ConllCorpusReader(config.classifier_predictions_folder, config.classifier_predictions_filename_no_folder, ['words', 'pos'])



		# A list of tokens and the method used to normalise them. Intented to hold tuples:
		# (tag, prediction_source, word, prediction, correct_word, correct?)
		normalised_tokens = []
		normalised_documents = []

		total = 0
		total_correct = 0

		for i, doc in enumerate(corpus.tagged_sents()):
			# Ignore if all tags are "O"
			#if not len([d[1] for d in doc if d[1] != "O"]) == 0:
			#print i, sent
			test_doc = test_data[i]["input"]


			normalised_documents.append([])
			normalised_documents[-1].append(test_data[i]["index"])
			normalised_documents[-1].append([])


			#print "INDEX: %s" % test_data[i]["index"]

			for j, (word, tag) in enumerate(doc):


				original_word = test_data[i]["input"][j]

				word = test_data[i]["input"][j].lower()
				correct_w = truth_data[i]["output"][j]		

				#if word in unique_replacements and word.isalpha():
				#	print "UNQ".ljust(5), Fore.YELLOW + word.ljust(30) + Style.RESET_ALL, word.ljust(30), truth_data[i]["output"][j].ljust(30), (Fore.GREEN if word == truth_data[i]["output"][j] else  Fore.RED) + str(word == truth_data[i]["output"][j]) + Style.RESET_ALL

				pred_source = None
				pred = word


				if word in unique_replacements:
					pred = unique_replacements[word]
					pred_source = M_DICTIONARY_NORMALISATION

				if word.isalpha() and word not in self.word_set:
				#if word not in unique_replacements and word[0] != "@" and word[0] != "#" and word[0].islower(): (twitter version)
				

					# The dictionary normalisation has the highest priority.
					if word not in unique_replacements:
						#if config.classification_strategy == NONE:
						#	tag = "SPE"						
						# If it is a spelling error (or deemed to be 'correct' by the classifier, despite not being), normalise it using the chosen spellchecking strategy.
						if tag == "SPE" or tag == "O":
							pred, pred_source = spellingErrorNormaliser.normalise(j, document = [s.lower() for s in test_doc], document_index = i)					
						# If it is an acronym, normalise it via regex.
						elif tag == "ACR":
							pred, pred_source = acronymNormaliser.normalise(word)
						# If it is a DST, the prediction will be the input word.
						elif tag == "DST":
							pred = word 
							pred_source = M_DST
						#elif tag == "O":
						#	pred = word
						#	pred_source = "Ignored"


					prediction_data[i]["output"][j] = pred					
					is_correct = correct_w == pred
					if is_correct:
						total_correct += 1
					total += 1	
					if self.verbose:					
						print tag.ljust(5), original_word.ljust(30), pred.ljust(30), correct_w.ljust(30), (Fore.GREEN if is_correct else  Fore.RED) + str(is_correct) + Style.RESET_ALL

					correct_tag = get_word_class(original_word, correct_w)
					tag_is_correct = tag == correct_tag

					normalised_tokens.append((tag, pred_source, original_word, pred, correct_w, is_correct, correct_tag, tag_is_correct))

					normalised_documents[-1][-1].append([tag, pred_source, original_word, pred, correct_w, is_correct, correct_tag, tag_is_correct])

				else:
					normalised_documents[-1][-1].append(word)
					

		print "Total: %d / %d (%.2f%%)" % (total_correct, total, 100.0 * total_correct/total)
		return prediction_data, normalised_tokens, normalised_documents


	def save_predictions_to_file(self, prediction_data, truth_data):
		with open(config.predictions_output_filename, 'w') as f:
			json.dump(prediction_data, f)
		with open(config.truth_output_filename, 'w') as f:
			json.dump(truth_data, f)		


	def save_normalised_documents_to_file(self, normalised_documents):
		with open(config.normalised_documents_output_filename, 'w') as f:
			json.dump(normalised_documents, f)

	def save_normalised_tokens_to_file(self, normalised_tokens):
		with open(config.normalised_tokens_output_filename, 'w') as f:
			json.dump(normalised_tokens, f, indent=4)


if __name__ == "__main__":
	LexnormPipeline()