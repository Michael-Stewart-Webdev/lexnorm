import os
import json
import shutil
import codecs


directory = "output"
dataset_folders = [x for x in os.listdir(directory)]



all_results = {}


for df in dataset_folders:
	experiments = [x for x in os.listdir(directory + "/" + df)]

	all_results[df] = {}

	for e in experiments:
		all_results[df][e] = {}
		result_folder = directory + "/" + df + "/" + e 

		normalised_tokens = json.loads(codecs.open(result_folder + "/normalised_tokens.json", 'r', 'utf-8').read())
		scores            = json.loads(codecs.open(result_folder + "/results.json", 'r', 'utf-8').read())

		all_results[df][e]["normalised_tokens"] = normalised_tokens
		all_results[df][e]["scores"] = scores
		

print all_results
with open("visualisation/data/results.js", 'w') as f:
	f.write("results = ")
	json.dump(all_results, f)