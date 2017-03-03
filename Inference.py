
import pickle
import numpy as np
from Fuzzification import fuzzify
from Defuzzification import defuzzify

def infer(data, rules, centroids, overlap, mf, target_centroids):
	'''
	Inputs: 
		data: nummpy array of size > number of centroids x 2
		rules: integer Rule Base as list if lists
		centroids: either a integer (for each feature te same)
	          or an array size = number of features
		mf:  'triangle', 'trapezoid' or 'Gaussian'
		overlap: number between 0 and 1, 
					when gaussian mf overlap is variance
					when triangle/trapezoid overlap is half of the base
		target_centroids: the cenroids of the target feature sets, 
				needed for the defuzzification

	Output: crisp output
	'''
	# collect crisp output
	crisp_output = []
	i = 0
	N = len(data)
	mean_firings = []
	nots = 0
	fire_list = []
	for x in data:
		print(i/N*100)
		# first clenollect fuzzy output
		fuzzy_y = []
		fuzzy_x = fuzzify(x, centroids, overlap, mf)
		for rule in rules:
			# for each index get memberships of set at that index 
			# and then the ancedent membership at that index  
			# take the min
			# create list like [consequent, firing strength]
			firing = min([fuzzy_x[i][rule[i]-1] for i in range(len(x))])
			# fuzzy_y = [consequent set, firing strength]
			fuzzy_y.append([rule[-1], firing])
			fire_list.append(firing)
		mean_firings.append(np.mean(fire_list))
		# then defuzzify fuzzy output
		crisp, not_fired = defuzzify(fuzzy_y, target_centroids)
		nots += not_fired
		crisp_output.append(crisp)

		i += 1
	pickle.dump(fire_list, open('firings_e1.p', 'wb'))
	print('nots', nots/len(data), len(data))
	return(crisp_output)


		


