import random, pickle

N_VARIANTS = 6
WIN_SIZE   = 10
ALPHABET   = "abcdefghijklmnopqrstuvwxyz_"
MAX_WORD_SET_SIZE = 300

def generateSampleData():

	

	# Get the set of valid words
	with open("data/google-10000-english-no-swears.txt", "r") as f:
	  lines = f.read().splitlines() 
	WORD_SET = set(lines)
	short_words_set = set()
	for w in WORD_SET:
		if len(w) <= WIN_SIZE:
			if len(short_words_set) < MAX_WORD_SET_SIZE:
				short_words_set.add(w)

	output = []

	for i in range(N_VARIANTS):
		for w in short_words_set:
			word = w		
			r = random.randrange(0, len(w))
			corrupted_word = w[:r] + random.sample(ALPHABET, 1)[0] + w[r+1:]

			output.append((word.ljust(WIN_SIZE, "_"), corrupted_word.ljust(WIN_SIZE, "_")))


	# Print the data (human-readable)
	with open("sample_data.txt", "w") as f:
		for line in output:
			f.write("%s\t%s" % (line[0], line[1]))
			f.write("\n")

	# Encode the data (for the NN)
	def encode_y(word):
		encoding = [[0 for a in range(len(ALPHABET))] for b in range(WIN_SIZE)]
		c = 0
		for char in word:
			index = ALPHABET.index(char)
			encoding[c][index] = 1
			c += 1
		return encoding
	def encode_x(word):
		return [ALPHABET.index(c) for c in word]

	nn_output = []
	for pair in output:
		y = encode_y(pair[0])
		x = encode_x(pair[1])
		nn_output.append((x, y))

	# Print the data (human-readable)
	with open("sample_data_nn.txt", "w") as f:
		for line in nn_output:
			f.write("%s\t%s" % (line[0], line[1]))
			f.write("\n")	

	with open("sample_data_pickle.pkl", "w") as f:
		pickle.dump(nn_output, f)

if __name__ == '__main__':
	generateSampleData()