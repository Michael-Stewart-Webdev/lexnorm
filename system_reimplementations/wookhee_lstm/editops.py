from Levenshtein import editops

a = "dog"
b = "cats"
print a
print b
print '---'

def process_editops(eds, t1, t2):
	s = ""
	ops = ["none"] * len(t2)
	counter = 0
	for ed in eds:		
		ops[ed[2]] = (ed[0] + "_" + b[ed[2]])
		counter += 1
		return tuple(ops)


eds = editops(a, b)
ops = process_editops(eds, a, b)

print ops