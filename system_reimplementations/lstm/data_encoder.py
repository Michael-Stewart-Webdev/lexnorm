from utils import *

# Encodes the training and testing data into a format that may be learnt and evaluated by the LSTM.
# Training data: the training data
# Testing data: the testing data
# dsts_training: A list of domain-specific terms found in the training data
# acronyms_training: A list of acronyms found in the training data
# dsts: All domain-specific terms
# acronyms: All acronyms
def encode_data(training_data, testing_data, dsts_training, replacements_training, acronyms_training, dsts, acronyms, word_set, fold = 1, context_window=3):
	
	def encode(data, dst_set, acronym_set, mode="training"):
		print ""

		# Removes any non-alphabetical characters from words.
		def clean_word(word):
			return ''.join([i if i.isalpha() else '' for i in word])

		# Converts a sequence to a character index sequence.
		def convert_to_char_indexes(sequence):
			indexes = [alphabet.find(c) for c in sequence]
			return indexes

			# Get the frequencies of all tokens in the data.
		def get_token_frequencies():			
			token_frequencies = defaultdict(int)
			total_words = 0
			for document in data:
				for token in document["input"]:
					if token.isalnum():
						total_words += 1
						token_frequencies[token.lower()] += 1
			with open(TOKEN_FREQUENCIES_FILE, "w") as f:
				f.write("Total words    : %s\n======================\n" % total_words)
				for k in sorted(token_frequencies, key = token_frequencies.get, reverse=True):
					f.write("%s %s\n" % (k.ljust(16), token_frequencies[k]))
			return token_frequencies, total_words

		error_type_frequencies = defaultdict(int)
		token_frequencies, total_words = get_token_frequencies()
		token_count    = 0
		document_count = 0

		lstm_input_x = []
		lstm_input_y = []		
		# Maintain a list of token indexes which are 0 if the word doesn't need to be classified, 1 if it does
		token_indexes = [] 
		# Maintain a list of token frequencies, to help compute the frequency clusters
		token_frequencies_list = []

		print "Encoding %s data..." % mode
		# Iterate over every document in the data and encode it into character indexes.
		doc_id = 0
		for document in data:	
			#tok_id = 0		
			# Add start and end of sentence chars
			#document["input"].insert(0, START_OF_SENTENCE_CHAR)
			#document["input"].append(END_OF_SENTENCE_CHAR)
			#document["output"].insert(0, START_OF_SENTENCE_CHAR)
			#document["output"].append(END_OF_SENTENCE_CHAR)



			for i in range(len(document["input"])):				

				error_type = NO_ERROR

				if document["input"][i].isalpha():

					inp = document["input"][i].lower()
					outp = document["output"][i].lower()
					#print inp, outp

					in_lexicon = 0
					if inp.lower() in word_set:
						in_lexicon = 1
					
					error_type = NO_ERROR

					# Figure out the error type of the word if there is one
					if inp != outp:
						error_type = SPELLING_ERROR					# Spelling error if input doesn't match output		
					else:
						if inp in dst_set:
							error_type = DOMAIN_SPECIFIC			# Domain-specific if output is not in lexicon, and is the same in the
																	# original and annotated data
																	# "bogger" -> "bogger", for example.
					if inp in acronym_set:
						error_type = ACRONYM 						# Acronym if input is part of pre-defined list of acronyms

					if i > 2:
						pw2 = clean_word(document["input"][i-2])
					else:
						pw2 = START_OF_SENTENCE_CHAR
					if i > 1:
						pw = clean_word(document["input"][i-1])
					else:
						pw = START_OF_SENTENCE_CHAR						
					if i < len(document["input"]) - 1:
						nw = clean_word(document["input"][i+1])
					else:
						nw = END_OF_SENTENCE_CHAR
					if i < len(document["input"]) - 2:
						nw2 = clean_word(document["input"][i+2])
					else:						
						nw2 = END_OF_SENTENCE_CHAR
					#if document["input"][i-1] == START_OF_SENTENCE_CHAR:
					#	pw = START_OF_SENTENCE_CHAR
					#if document["input"][i+1] == END_OF_SENTENCE_CHAR:
					#	pw = END_OF_SENTENCE_CHAR		
					tw = clean_word(inp)

					token_frequency = (token_frequencies[inp] * 1.0 / total_words)
					token_frequencies_list.append(token_frequency)

					

					if context_window == 3:
						data_x_chars = pw2 + " " + pw  + " " + tw + " " + nw + " " + nw2
					elif context_window == 5:
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
						if error_type != NO_ERROR:
							data_x_chars = pw2 + " " + pw  + " " + tw + " " + nw + " " + nw2
							data_x = convert_to_char_indexes(data_x_chars) + [len(tw)]# + [token_frequency]
							data_y = errorInt(error_type)
							#token_indexes.append((doc_id, tok_id, inp))
							#print [doc_id, tok_id], inp
							lstm_input_x.append(data_x)
							lstm_input_y.append(data_y)
							error_type_frequencies[error_type] += 1
							token_indexes.append(1)		
						else:
							token_indexes.append(0)			
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

		#if not simple:
		print "\n\nComputing the k-means clusters for the frequencies of each %s sample..." % mode
		# Compute k-means clusters for the frequencies
		X = np.array(token_frequencies_list).reshape(-1, 1)
		kmeans = KMeans(n_clusters=FREQUENCY_CLUSTERS_COUNT, random_state=0).fit(X)
		kmeans_labels = kmeans.predict(X)

		# Add the labels to the data
		for i in range(len(lstm_input_x)):	
			lstm_input_x[i].append(int(kmeans_labels[i]))

		lstm_input = [lstm_input_x, lstm_input_y]

		lmli = len(lstm_input_x)
		lmliy = len(lstm_input_y)
		print "\nTotal number of %s pairs:" % mode, lmli, lmliy


		#if mode == "testing":
		#	for ti in token_indexes:
		#		print ti

		return lstm_input, token_indexes
			#teststr += "===\n"

	def save_data(training_data, testing_data):
		print "Saving the data..."

		if not os.path.exists("classifiers/lstm/data/fold %d" % fold):
			os.makedirs("classifiers/lstm/data/fold %s" % fold)

		with open("classifiers/lstm/data/fold %d/lstm_data_training.pkl" % fold, "w") as f:
			pickle.dump(training_data, f)

		with open("classifiers/lstm/data/fold %d/lstm_data_testing.pkl" % fold, "w") as f:
			pickle.dump(testing_data, f)

		# Print debug info
		if not os.path.exists("classifiers/lstm/debug/fold %d" % fold):
			os.makedirs("classifiers/lstm/debug/fold %d" % fold)

		with open("classifiers/lstm/debug/fold %d/training_x.txt" % fold, "w") as f:
			for row in training_data[0]:
				f.write(str(row))
				f.write("\n")
		with open("classifiers/lstm/debug/fold %d/training_y.txt" % fold, "w") as f:
			f.write(str(training_data[1]))
		with open("classifiers/lstm/debug/fold %d/testing_x.txt" % fold, "w") as f:
			for row in testing_data[0]:
				f.write(str(row))
				f.write("\n")
		with open("classifiers/lstm/debug/fold %d/testing_y.txt" % fold, "w") as f:
			f.write(str(testing_data[1]))

		return ("classifiers/lstm/data/fold %d/lstm_data_training.pkl" % fold, "classifiers/lstm/data/fold %d/lstm_data_testing.pkl" % fold)







	lstm_training_data, lstm_training_token_indexes = encode(training_data, dsts, acronyms, mode="training")
	lstm_testing_data, lstm_testing_token_indexes  = encode(testing_data, dsts, acronyms, mode="testing")

	training_filename, testing_filename = save_data(lstm_training_data, lstm_testing_data)


	return (lstm_training_data, lstm_testing_data, lstm_training_token_indexes, lstm_testing_token_indexes, training_filename, testing_filename)








