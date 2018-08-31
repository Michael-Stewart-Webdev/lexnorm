import json

from collections import defaultdict
import pickle
import numpy as np
import shutil, os

import random, pickle


from sklearn.cluster import KMeans
import numpy as np


ANNOTATED_DATA_FILE     = "../../data/annotated_data.json"
ML_INPUT_HR_DATA_FILE   = "debug/ml_input_humanreadable.txt"
TOKEN_FREQUENCIES_FILE  = "debug/token_frequencies.txt"
ACRONYM_REPLACEMENTS_FILE = "../../data/acronyms.txt"

LSTM_DATA_TR_OUTPUT_FILE   = "data/lstm_data_training.pkl"
LSTM_DATA_TE_OUTPUT_FILE   = "data/lstm_data_testing.pkl"

NO_ERROR       		= "NO_ERROR"			
SPELLING_ERROR 		= "SPELLING_ERROR"	
ACRONYM 	   		= "ACRONYM"			
ABBREVIATION   		= "ABBREVIATION"	
DOMAIN_SPECIFIC     = "DOMAIN_SPECIFIC"	

START_OF_SENTENCE_CHAR = "<"
END_OF_SENTENCE_CHAR   = ">"

#alphabet = "abcdefghijklmnopqrstuvwxyz1234567890.'\"~!@#$%^&*(),:;|\\/ <>"
alphabet = "abcdefghijklmnopqrstuvwxyz <>"

FREQUENCY_CLUSTERS_COUNT = 4 # The number of frequency classes to append to the training inputs

# Returns an integer value of the error type
def errorInt(error_type):
	if error_type == SPELLING_ERROR:
		return 0
	elif error_type == DOMAIN_SPECIFIC:
		return 1
	elif error_type == ACRONYM:
		return 2
	elif error_type == ABBREVIATION:
		return 3
	else:		
		return -1

#with open("../../data/words_en.txt", "r") as f:
#	lines = f.read().splitlines() 

#WORD_SET = set(lines)


#acronyms = []
#for line in open(ACRONYM_REPLACEMENTS_FILE, "r"):
#	line = line.replace("\n", "")
#	split = line.split("\t")		
#	acronyms.append(split[0])


#ACRONYM_SET  	 = set([a.lower() for a in acronyms])