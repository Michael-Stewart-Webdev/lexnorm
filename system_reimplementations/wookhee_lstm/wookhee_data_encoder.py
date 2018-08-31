from utils import *
from Levenshtein import editops
from collections import Counter
import operator
from collections import defaultdict

tokens_dict = defaultdict(list)

# Encodes the training and testing data into a format that may be learnt and evaluated by the LSTM.
# Training data: the training data
# Testing data: the testing data
# dsts_training: A list of domain-specific terms found in the training data
# acronyms_training: A list of acronyms found in the training data
# dsts: All domain-specific terms
# acronyms: All acronyms
def encode_data(training_data, testing_data, dsts_training, replacements_training, acronyms_training, dsts, acronyms, word_set, fold = 1, context_window=3):
	

	# Determine the edit operations required to convert token1 into token2
	def edit_operations(token1, token2):
		eds = editops(token1, token2)
		s = ""
		ops = ["none"] * max(len(token2), len(token1))
		counter = 0
		
		for ed in eds:		
			if ed[0] == "insert" or ed[0] == "replace":
				ops[ed[2]] = (ed[0] + "_" + token2[ed[2]])
			elif ed[0] == "delete":
				ops[ed[2]] = (ed[0] + "_" + token1[ed[2]])
			counter += 1

		# Remove trailing "none"s
		#if token1 == "breakbased":
		#	print ops
		#	print "==="
		for x in reversed(range(len(ops))):
			if ops[x] == "none":
				del ops[x]
			else:
				break


		#if token2 == "emergency management officer":
		#	print "EMERGENCY MANAGEMENT OFFICER\n====================="
		#	print token1
		#	print token2
		#	print ops 
		#	print "==========================="

		'''
		if ops != []:
			print "<<"
			print "Tokens:      " + token1, token2
			print "Eds:         " + str(eds) 
			print "Operations:  " + str(tuple(ops))
			print ">>"
		'''

		return tuple(ops)



	def encode(data, dst_set, acronym_set, edit_operations_dict, mode="training"):
		print ""
		#print edit_operations_dict

		# Removes any non-alphabetical characters from words.
		def clean_word(word):
			return ''.join([i if i.isalpha() else '' for i in word])

		# Converts a sequence to a character index sequence.
		def convert_to_char_indexes(sequence):
			indexes = [alphabet.find(c) for c in sequence]
			return indexes

		error_type_frequencies = defaultdict(int)
		token_count    = 0
		document_count = 0

		lstm_input_x = []
		lstm_input_y = []	

		#lstm_input_y_ops = []	
		# Maintain a list of token indexes which are 0 if the word doesn't need to be classified, 1 if it does
		token_indexes = [] 
		# Maintain a list of token frequencies, to help compute the frequency clusters

		global tokens_dict
		


		print "Encoding %s data..." % mode
		# Iterate over every document in the data and encode it into character indexes.
		doc_id = 0
		for document in data:	

			for i in range(len(document["input"])):				

				error_type = NO_ERROR

				if document["input"][i].isalpha() and document["input"][i] not in word_set:

					inp = document["input"][i].lower()
					outp = document["output"][i].lower()
					#print inp, outp

					in_lexicon = 0
					if inp.lower() in word_set:
						in_lexicon = 1
					
					if i > 1:
						pw = clean_word(document["input"][i-1])
					else:
						pw = START_OF_SENTENCE_CHAR						
					if i < len(document["input"]) - 1:
						nw = clean_word(document["input"][i+1])
					else:
						nw = END_OF_SENTENCE_CHAR

					#if document["input"][i-1] == START_OF_SENTENCE_CHAR:
					#	pw = START_OF_SENTENCE_CHAR
					#if document["input"][i+1] == END_OF_SENTENCE_CHAR:
					#	pw = END_OF_SENTENCE_CHAR		
					tw = clean_word(inp)


					

					data_x_chars = pw  + " " + tw + " " + nw

					#print data_x_chars

					can_add = False
					if mode == "testing":
						# Don't add to the test data if it's part of the dsts_training, replacements_training, or acronyms_training
						# Only need to test on tokens that weren't seen in the training data.
						if inp in dsts_training or inp in acronyms_training or inp in replacements_training:	
							can_add = False
						else:
							can_add = True
					else:
						can_add = True

					if can_add:
						
						data_x_chars = pw  + " " + tw + " " + nw
						data_x = convert_to_char_indexes(data_x_chars)# add POS tag later
						eds = edit_operations(inp, outp)
						
						if eds not in edit_operations_dict:
							if mode == "training": # Only add a new edit operation to the dictionary in training mode
								#data_y = eds
								#print eds, len(edit_operations_dict)
								#print "--"
								edit_operations_dict[eds] = len(edit_operations_dict)	
							elif mode == "testing":
								eds = ()


							#else:
							#	data_y = 0 # If the edit operations have not been seen before (in testing mode), assume
											# that there are no operations to perform
											# This makes more sense than adding an operation to the dictionary again,
											# as the operations should only be based on those found in the training data
						#else:
							#data_y = eds
							#if mode == "testing":
							#	print data_y, inp, outp
						#print data_y

						#print edit_operations_dict[eds]
						data_y = edit_operations_dict[eds]
						tokens_dict[edit_operations_dict[eds]].append(inp)
						#print tokens_dict[edit_operations_dict[eds]]
						

						lstm_input_x.append(data_x)
						lstm_input_y.append(data_y)
						token_indexes.append(1)		
	
					#if simple:
					#	data_x = [len(tw), int(token_frequency * 100)]

					# If it's not an acronym and has a length of 1, ignore it
					#if len(inp) == 1 and error_type != ACRONYM:
					#	error_type = NO_ERROR


					else:
						token_indexes.append(0)
				else:
					token_indexes.append(0)
				#tok_id += 1
			#doc_id += 1	

			document_count += 1

			if document_count % 100 == 0:
				print document_count,


		#print "\nTOKEN INDEXES: ", len(token_indexes)


		# Add the labels to the data
		#for i in range(len(lstm_input_x)):	
		#	lstm_input_x[i].append(int(kmeans_labels[i]))

		#edit_operations_list = tuple(edit_operations_set)
		print "==="
		print str(len(edit_operations_dict)) + " unique edit operations in total"

		#if mode == "training":
		with open("grr.txt", "w") as f:
			for x in sorted(edit_operations_dict.items(), key=operator.itemgetter(1)):
				f.write(str(x[1]) + "\t" + str(x[0]) + "\n")
				if len(tokens_dict[x[1]]) == 0:
					print "BROKEN 000000000000000000000000000000000000000000000000000000000000000"
				f.write(str(tokens_dict[x[1]]))
				f.write("\n")

		#lstm_input_y = [None] * len(lstm_input_x)
 		#for i in range(len(lstm_input_x)):	
			#if edit_operations_dict[lstm_input_y[i]] == 0:
			#	print lstm_input_y[i], edit_operations_dict[lstm_input_y[i]]
			#	print "---"
		#	lstm_input_y[i] = edit_operations_dict[lstm_input_y_ops[i]]	# Convert operations to a class label


		lstm_input = [lstm_input_x, lstm_input_y]

		print "Most common operations: "
		c = Counter(lstm_input_y)
		print c.most_common(1000)

		lmli = len(lstm_input_x)
		lmliy = len(lstm_input_y)
		print "\nTotal number of %s pairs:" % mode, lmli, lmliy


		#if mode == "testing":
		#	for ti in token_indexes:
		#		print ti

		return lstm_input, token_indexes, edit_operations_dict
			#teststr += "===\n"

	def save_data(training_data, testing_data):
		print "Saving the data..."

		if not os.path.exists("classifiers/wookhee_lstm/data/fold %d" % fold):
			os.makedirs("classifiers/wookhee_lstm/data/fold %s" % fold)

		with open("classifiers/wookhee_lstm/data/fold %d/lstm_data_training.pkl" % fold, "w") as f:
			pickle.dump(training_data, f)

		with open("classifiers/wookhee_lstm/data/fold %d/lstm_data_testing.pkl" % fold, "w") as f:
			pickle.dump(testing_data, f)

		# Print debug info
		if not os.path.exists("classifiers/wookhee_lstm/debug/fold %d" % fold):
			os.makedirs("classifiers/wookhee_lstm/debug/fold %d" % fold)

		with open("classifiers/wookhee_lstm/debug/fold %d/training_x.txt" % fold, "w") as f:
			for row in training_data[0]:
				f.write(str(row))
				f.write("\n")
		with open("classifiers/wookhee_lstm/debug/fold %d/training_y.txt" % fold, "w") as f:
			f.write(str(training_data[1]))
		with open("classifiers/wookhee_lstm/debug/fold %d/testing_x.txt" % fold, "w") as f:
			for row in testing_data[0]:
				f.write(str(row))
				f.write("\n")
		with open("classifiers/wookhee_lstm/debug/fold %d/testing_y.txt" % fold, "w") as f:
			f.write(str(testing_data[1]))

		return ("classifiers/wookhee_lstm/data/fold %d/lstm_data_training.pkl" % fold, "classifiers/wookhee_lstm/data/fold %d/lstm_data_testing.pkl" % fold)





	#edit_operations_dict = {(): 0}
	edit_operations_dict = {}

	lstm_training_data, lstm_training_token_indexes, edit_operations_dict1 = encode(training_data, dsts, acronyms, edit_operations_dict, mode="training")
	
	#print "L1", len(edit_operations_dict)

	lstm_testing_data, lstm_testing_token_indexes, edit_operations_dict2  = encode(testing_data, dsts, acronyms, edit_operations_dict1, mode="testing")

	#print "L2", len(edit_operations_dict)

	training_filename, testing_filename = save_data(lstm_training_data, lstm_testing_data)

	#for i in edit_operations_dict:
	#	if edit_operations_dict[i] == 7:
	#		print i

	return (lstm_training_data, lstm_testing_data, lstm_training_token_indexes, lstm_testing_token_indexes, training_filename, testing_filename, edit_operations_dict2)



# Decode the operations into a token.
def decode_operations(ops, token):
	# 'none', 'none', 'none', 'none', 'none', 'none', u'replace_i', 'none', 'none', 'none', u'insert_e'  << Example
	if ops == ():
		#print "(", token, " already correct )"
		return token
	converted_token = None
	token = token.ljust(len(ops), " ")

	token_ls = list(token)

	print ops, token,

	i = 0
	for op in ops:
		if op.startswith("insert_"):
			token_ls.insert(i, op[7])
			i += 1
		elif op.startswith("delete_"):
			token_ls[i] = ''
		elif op.startswith("replace_"):
			token_ls[i] = op[8]
		i += 1
	converted_token = ''.join(token_ls)
	#print ">>"
	#print "TOKEN:   ", token
	#print "F TOKEN: ",converted_token
	#print "OPS:     ", ops
	#print "<<"
	print converted_token
	return converted_token






