#########################################################
# Cooperative Rule (COR) Approach for making a
# Fuzzy Rule Base with the Wang and Mendel method
# Functions: learn
#########################################################


from Fuzzification import fuzzify
import membership_function
import numpy as np
import itertools
import random
from Inference import infer
from Defuzzification import defuzzify
from error_measures import RMSE


def learn(data, centroids, overlap, mf, target_col, min_y, max_y, iterations = 1):
	'''
	Inputs: 
		data: nummpy array of size > number of centroids x 2
		target_col: integer index of the target column
		Centroid: either a integer (for each feature te same)
	          or an array size = number of features
		mf:  'triangle', 'trapezoid' or 'Gaussian'
		overlap: number between 0 and 1, 
					when gaussian mf overlap is variance
					when triangle/trapezoid overlap is half of the base 
		iterations: how many iterations to test the COR with


	Output: integer Rule base = list of size rules x features
	'''
	RB = {}
	for x in data:
		# fuzzify the data point
		fuzzy_x = fuzzify(x, centroids, overlap, mf)
		# The rule is a combination of the sets with maximum membership
		rule = []
		degree = 1
		for fset in fuzzy_x:
			# the winner set (1-nsets) is the set with higest membership
			winner_set = np.argmax(fset)
			rule.append(winner_set+1)
			degree *= fset[winner_set]
		# key = rule, value = [all possible consequents]
		consq = rule[target_col]
		# delete consequent
		del rule[target_col]
		rule = tuple(rule)
		RB.setdefault(rule, []).append(consq)
	# before the inference, seperate the target values from the data
	target_centroids = centroids[target_col]
	centroids = np.delete(centroids,target_col, 0)
	targets = data[:, target_col]
	data = np.delete(data,target_col,1)
	# stochasticly optimize RB
	lowest_error = 9999
	for i in range(iterations):
		RB_sample = []
		# for each unique combination of ancedents
		for rule in RB:
			# randomly choose a consequent
			cons = random.choice(RB[rule]+['dc'])
			# ignore dont cares
			if not(cons == 'dc'):
				RB_sample.append(list(rule)+[cons])
		# start inference 
		crisp= infer(data, RB_sample, centroids, overlap, mf, target_centroids)
		error =  RMSE(min_y, max_y, crisp, targets)
		if error < lowest_error:
			RB_best = RB_sample
	return(RB_best)




	




	
	
