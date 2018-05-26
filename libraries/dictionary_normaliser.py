'''
Created on 8 Mar, 2017

Normalises a token using a direct dictionary lookup, regardless of the context of the token.

@author: michaelstewart
'''

import enchant
import re
from collections import Counter
import editdistance
#import shutil
import os

class DictionaryNormaliser:	   

	def __init__(self, pwl_filename):

		self.d = enchant.DictWithPWL("en_GB", pwl_filename)

		return None

	def check(self, word):
		return self.d.check(word)

	def add_to_dictionary(self, word):
		self.d.add(word)

	def normalise(self, token):
		suggestions = self.d.suggest(token)

		if len(suggestions) > 1:
			try:
				suggestion = next(s for s in suggestions if s.isalpha()).lower()#suggestions[0]
			except StopIteration:
				suggestion = suggestions[0]
		else:
			suggestion = token.lower()

		return suggestion

