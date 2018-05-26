from config import Config
from utils import *
import sys


config = None

from global_vars import *

class AcronymNormaliser():


	def __init__(self, conf):
		global config
		config = conf if conf else Config()
		self.acronym_map = load_dataset(config.acronyms_filename)
		
	def set_config(self, conf):
		global config
		config = conf

	def normalise(self, word):
		return (self.acronym_map[word], M_ACRONYM) if word in self.acronym_map else (word, M_ACRONYM_ERROR)