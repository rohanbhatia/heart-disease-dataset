import numpy as np

def parse_data(input_file):
	#INPUT: An input file containing the data
	#OUTPUT: Returns a vector of inputs and a vector of targets

	inputs = list()
	targets = list()

	f = open(input_file)

	#parse each line
	for line in f:	
		
		example = line.split()
		
		count = 0
		while count < 14:
			example[count] = float(example[count])
			count += 1

		#heard disease dataset provides "degrees" of disease from 0 to 4, but we are only interested in abscence (0) or presence (1)
		if example[13] == 0.0:
			example[13] = 0
		else:
			example[13] = 1

		#add data example and it's target to the main input/target arrays
		inputs.append(example[0:13])
		targets.append(example[13])
	
	f.close()

	X = np.array(inputs)
	T = np.array(targets)

	return X, T

if __name__ == '__main__':
	pass
	