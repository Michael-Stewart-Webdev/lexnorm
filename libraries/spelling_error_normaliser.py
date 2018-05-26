from config import Config
import codecs
from libraries.dictionary_normaliser import DictionaryNormaliser
import gensim, difflib
from utils import *
from colorama import Fore, Style
import sys

sys.path.append("../")
from global_vars import *

config = None
		
WINDOW_SIZE = 2 # Number of words before and after word

class SpellingErrorNormaliser():

	def __init__(self, conf):
		global config
		config = conf if conf else Config()

		self.stopwords = load_wordset(config.stopwords_file)

		self.strategy = config.spellcheck_strategy
		self.dictionaryNormaliser = DictionaryNormaliser(config.aspell_pwl_file)
		print "Loading embedding model...",
		self.embedding_model = gensim.models.Word2Vec.load(config.emb_model_filename)
		print " done."
		if config.spellcheck_strategy == QRNN_SPE:
			self.qrnn_predictions = self.unsegment_qrnn_predictions()

	def set_config(self, conf):
		global config
		config = conf


	# Get the prediction using the trained word2vec model.
	def emb_prediction(self, wi, sent):
		# Add the padding characters to the sentence.
		original_w = sent[wi]
		#sent = ["_SOS_"] + sent + ["_EOS_"]
		
		#print "SENT:", sent 

		#print original_w 

		context_b = [w for w in sent[:wi] if w not in self.stopwords and w.isalpha()]
		context_a = [w for w in sent[wi+1:] if w not in self.stopwords and w.isalpha()]
		wi = wi-(len(sent[:wi]) - len(context_b))
		context = context_b + [""] + context_a

		#print "Before:", context_b 
		#print "After:", context_a
		#print context 


		#context = [w for w in sent if w not in self.stopwords]#[sent[wi], sent[wi+2]]
		context = context[(wi-WINDOW_SIZE if wi- WINDOW_SIZE >= 0 else 0):wi] + context[wi+1:wi+WINDOW_SIZE+1]
		emb_predictions = self.embedding_model.predict_output_word(context, topn=50)


		#exit()
		# print ""
		# print '-' * 50

		# print "Word: ", original_w
		# print "Context: ", context
		# print '-' * 50 
		msw = []
		if emb_predictions:
			ms  = [m[0] for m in emb_predictions if m[0] not in context]
			msw = difflib.get_close_matches(original_w, ms, n=3, cutoff=0.8)
			# print "EMB predictions: " + ", ".join(sorted(ms))
		if len(msw) > 0:
			# print "Most similar:", msw			
			return msw[0]
		else:
			# print "(Not found)"
			return None


	# Un-segments the qrnn predictions using the segment metadata file, and turns them into the same format as the 
	# original JSON test set.
	def unsegment_qrnn_predictions(self):
		print "Unsegmenting QRNN predictions..."
		with codecs.open(config.qrnn_predictions_file, 'r', 'utf-8') as f:
			qrnn_predictions = [line.strip().split() for line in f]
		with codecs.open(config.qrnn_test_input_file_all, 'r', 'utf-8') as f:
			original_inputs = [line.strip().split() for line in f]

		segment_metadata = load_dataset(config.qrnn_segmentation_metadata_file)
		output_data = []
		combined = []
		for i, (sent, ending) in enumerate(zip(qrnn_predictions, segment_metadata)):
			#print i, "\t", ending, "\t", sent
			#if i > 25:
			#	raw_input()

			# If the sentence was too long, replace it with the original input
			if sent == [config.qrnn_sent_too_long_char]:
				sent = original_inputs[i]

			if ending in [".N", ",N", "N"]:
				if len(ending) > 1:
					combined += sent + [ending[0]]
				else:
					combined += sent
				output_data.append(combined)
				combined = []
			else:
				combined += sent + [ending]


		with codecs.open(config.qrnn_predictions_file + '2', 'w', 'utf-8') as f:
			f.write("\n".join([" ".join(s) for s in output_data]))
		return output_data


	# Normalise a word.
	# The inputs are the index of the word, and the sentence.
	# The output is the normalised word, and the method used to normalise it.
	def normalise(self, word_index, document=None, document_index=None):



		word = document[word_index]

		if config.spellcheck_strategy == QRNN_SPE:
			# TODO: Make this work properly

			#print document_index
			#print "DOC"
			#print ' '.join([Fore.YELLOW + "[" +  document[w] + "]" + Style.RESET_ALL if w == word_index else document[w] for w in range(len(document))])
			try:
				#print "QRNN PRED"
				#print "Word index: ", word_index
				#print ' '.join([Fore.YELLOW + "[" +  self.qrnn_predictions[document_index][w] + "]" + Style.RESET_ALL if w == word_index else self.qrnn_predictions[document_index][w] for w in range(len(self.qrnn_predictions[document_index]))])
				#print "QRNN PRED FOR WORD:"
				#print self.qrnn_predictions[document_index][word_index]
				return self.qrnn_predictions[document_index][word_index].replace("_", " "), M_QRNN
			except IndexError:
				#print Fore.RED + word + Style.RESET_ALL
				return word, M_QRNN_ALIGNMENT_ERROR
			print ""
		elif config.spellcheck_strategy == EMB_ONLY:
			emb_pred = self.emb_prediction(word_index, document)
			if emb_pred == None:
				return word, M_EMB_FAILED
			else:
				return emb_pred, M_EMB

		elif config.spellcheck_strategy == EMB_ASPELL:

			emb_pred = self.emb_prediction(word_index, document)
			if emb_pred == None:

				return self.dictionaryNormaliser.normalise(word), M_ASPELL
			else:
				return emb_pred, M_EMB
		elif config.spellcheck_strategy == ASPELL_ONLY:
			return self.dictionaryNormaliser.normalise(word), M_ASPELL

		else:
			return word, M_NO_SPELLCHECKING