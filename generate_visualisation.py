import os
import json
import shutil
import codecs

from collections import OrderedDict
from config import Config
    

class VisualisationGenerator():

	def __init__(self, conf=None):

		config = conf if conf else Config()

		directory = config.output_path
		dataset_folders = [x for x in os.listdir(directory)]

		all_results = {}



		for df in dataset_folders:
			if os.path.isdir(directory + df):

				try:
					experiments = [x for x in os.listdir(directory + df)]


					df2 = df.split('_')[0] + "_0" + df.split('_')[1] if len(df.split('_')[1]) == 1 else df

					all_results[df2] = OrderedDict()

					for e in sorted(experiments):

						print e
						all_results[df2][e] = {}
						result_folder = directory + df + "/" + e 

						normalised_tokens 	 = json.loads(codecs.open(result_folder + "/normalised_tokens.json", 'r', 'utf-8').read())
						normalised_documents = json.loads(codecs.open(result_folder + "/normalised_documents.json", 'r', 'utf-8').read())
						scores           	 = json.loads(codecs.open(result_folder + "/results.json", 'r', 'utf-8').read())

						cor = sum([1 if x[5] else 0 for x in normalised_tokens]) # x[5] should be whether the token is correct or not
						incor = len(normalised_tokens) - cor

						all_results[df2][e]["normalised_tokens"] = normalised_tokens
						all_results[df2][e]["normalised_documents"] = normalised_documents
						all_results[df2][e]["scores"] = scores
						all_results[df2][e]["scores"]["Correct"] = cor
						all_results[df2][e]["scores"]["Incorrect"] = incor
				except IOError as e:
					print e


		o = OrderedDict()

		for k in sorted(all_results):
			o[k] = all_results[k]


		#print all_results
		with open(config.visualisation_folder + "data/results.js", 'w') as f:
			f.write("results = ")
			json.dump(o, f)


if __name__ == "__main__":
	VisualisationGenerator()

