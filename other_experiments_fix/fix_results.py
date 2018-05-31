import os
import json
import shutil
import re
import codecs

rootdir = os.getcwd()

experiments = [x for x in os.listdir(os.getcwd())]

for e in experiments:
	if e == "fixed":
		continue
	if os.path.isdir(e):
		folds = [x for x in os.listdir(e)]
		for fold in folds:
			
			with open("../data/" + fold.replace(" ", "_") + "/test_data.json") as f:
				test_data = json.loads(f.read())

			with open("../asset/" + fold.replace(" ", "_") + "/unique_replacements/unique_replacements.json") as f:
				unique_replacements = json.loads(f.read())

			predictions_file = e + "/" + fold + "/predictions.js"
			with open(predictions_file) as f:
				data_1 = f.read()

			print fold
			print ""
			data_2 = data_1[data_1.find(" = ")+2:]

			data = json.loads(data_2)["docs"]
			

			# Remove token types

			new_predictions = []

			#if fold != "fold 1" or e != "deep_encoding_dictionary":
			#	continue


			for i, doc in enumerate(data):

				doc = doc["tokens"]


				doc = doc.replace('<span class="token-type">&#10004;</span>', '')
				doc = doc.replace('<span class="token-type">&#x2717;</span>', '')
				doc = doc.replace('injured person\'s', 'ips')
				doc = doc.replace('nonsteroidal anti-inflammatory drugs', 'nsaids')
				doc = doc.replace("\n", " ")

				doc_split = re.split(r'<span ([^<]*)</span>', doc)



				doc_clean = []

				for d in doc_split:
					# print d
					#print d
					if "data-original" in d or "data-correction" in d:
						doc_clean.append(d[d.find("\">")+2:])
					else:
						for x in d.split():
							doc_clean.append(x)


				# if len(doc_clean) != len(test_data[i]["input"]):
				# 	print test_data[i]["index"]
				# 	print doc_clean
				# 	print "\n"
				# 	print len(doc_clean), len(test_data[i]["input"])

				# 	print ""
				# 	print doc_split

				# 	print ""
				# 	print fold, e
				# 	for x in range(len(doc_clean)):
				# 		print doc_clean[x], test_data[i]["input"][x]


				# 	exit()

				# if i == 1:
				# 	print doc_clean

				for x in range(len(doc_clean)):
					if doc_clean[x] in unique_replacements:
						#print "Replaced %s with %s" % (doc_clean[x], unique_replacements[doc_clean[x]])
						doc_clean[x] = unique_replacements[doc_clean[x]]
						
				new_predictions.append({})
				new_predictions[-1]["input"] = test_data[i]["input"]
				new_predictions[-1]["output"] = doc_clean
				new_predictions[-1]["index"] = test_data[i]["index"]

				#for x in range(len(new_predictions[-1]["input"])):
				#	print new_predictions[-1]["input"][x], new_predictions[-1]["output"][x]

				#if i > 0:
				#	exit()


			if not os.path.exists("fixed/" + e):
				os.mkdir("fixed/" + e)
			if not os.path.exists("fixed/" + e + "/" + fold):
				os.mkdir("fixed/" + e + "/" + fold)

			with open("fixed/" + e + "/" + fold + "/predictions.json", 'w') as f:
				json.dump(new_predictions, f)



		print ""