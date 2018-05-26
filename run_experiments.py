from process_data import Preprocessor
from run_pipeline import LexnormPipeline
from config import Config
from global_vars import *
from evaluation import evaluate
import os, shutil

from libraries.spelling_error_normaliser import SpellingErrorNormaliser
from libraries.acronym_normaliser import AcronymNormaliser


REMOVE_OLD_OUTPUT = True

class ExperimentRunner():

	def __init__(self):
		self.experiments = [
			 'fold_1'
			 #'fold_5'
			 #"twitter"
			 #"fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6", "fold_7", "fold_8", "fold_9", "fold_10"
			 #"fold_5", "fold_6", "fold_7", "fold_8", "fold_9", "fold_10"

			 #"fold_1", "fold_2", 
			 #"fold_8", "fold_9"#, "fold_10"
			 #"fold_10"
		]
		self.run_experiments()

	def delete_old_output(self, config):
		if os.path.isdir(config.output_path_exp):
			#if raw_input("Safe to delete all old outputs in %s? > " % config.output_path_exp) in ["y", "Y"]:
			if REMOVE_OLD_OUTPUT:
				shutil.rmtree(config.output_path_exp)
				print "All old outputs removed."

	def run_experiments(self):

		for e in self.experiments:
			i = 0
			print "=" * 100
			print "Running preprocessor for %s" % e 
			print "=" * 100
			config = Config(e)
			self.delete_old_output(config)
			Preprocessor(config)#, reset_all = True)

			spellingErrorNormaliser = SpellingErrorNormaliser(config)
			acronymNormaliser = AcronymNormaliser(config)

			print "=" * 100
			print "Starting experiments for %s" % e 
			print "=" * 100

			for d in DICT_NORMALISATION_STRATEGIES:				
				for c in CLASSIFICATION_STRATEGIES:
					for s in SPELLCHECKING_STRATEGIES:
						i += 1			
						config = Config(e, d, c, s, i)
						config.print_config()
						spellingErrorNormaliser.set_config(config)	
						acronymNormaliser.set_config(config)	
						LexnormPipeline(config, False, spellingErrorNormaliser, acronymNormaliser)
						evaluate(config)

ExperimentRunner()