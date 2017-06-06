#
#
#
#
#



from Fuzzification import fuzzify
import membership_function
import numpy as np

def learn(data, centroids, overlap, mf, target_col):
	'''
	Inputs: 
		data: nummpy array of size > number of centroids x 2
		target_col: integer index of the target column
		centroids: either a integer (for each feature te same)
	          or an array size = number of features
		mf:  'triangle', 'trapezoid' or 'Gaussian'
		overlap: number between 0 and 1, 
					when gaussian mf overlap is variance
					when triangle/trapezoid overlap is half of the base 


	Output: integer Rule base = list of size rules x features
	'''
	RB_array = []
	RB = {}
	for x in data:
		# fuzzify the data point
		fuzzy_x = fuzzify(x, centroids, overlap, mf)
		# The rule is a combination of the sets with maximum membership
		rule = []
		degree = 1
		for fset in fuzzy_x:
			# the winner set (1-nsets) is the set with highest membership
			winner_set = np.argmax(fset)
			rule.append(winner_set+1)
			degree *= fset[winner_set]
		# key = rule, value = [consequent, degree]
		consq = [rule[target_col], degree]
		# delete consequent
		del rule[target_col]
		rule = tuple(rule)
		# if rule already exists
		if rule in RB:
			# get the rule values in dict
			degree_old = RB[rule][1]
			# if old degree lower then new degree, replace consequent and degree
			if degree_old < degree:
				RB[rule] = consq
		# if not yet exists, add rule
		else: 
			RB[rule] = consq
	RB_array = [list(rule)+[RB[rule][0]] for rule in RB]
	return(RB_array)
	