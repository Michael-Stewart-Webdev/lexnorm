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
		self.d = enchant.Dict("en_GB")#, pwl_filename)

	def check(self, word):
		return self.d.check(word)

	def add_to_dictionary(self, word):
		self.d.add(word)

	def normalise(self, token):
		suggestions = self.d.suggest(token)

		#print self.check(token)

		if len(suggestions) > 1:
			#try:
		#		suggestion = next(s for s in suggestions).lower()# if s.isalpha()).lower()#suggestions[0]
		#	except StopIteration:
			suggestion = suggestions[0].lower()
		else:
			suggestion = token.lower()

		return suggestion



def main():



	d = DictionaryNormaliser("../asset/fold_1/pwl/pwl.txt")

	while True:
		i = raw_input("> ")
		print d.normalise(i)

if __name__ == '__main__':
	main()