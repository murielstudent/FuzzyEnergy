
#########################################################
# Deffuzzification of fuzzy ouputs
#
# Functions: defuzzify
#########################################################

import numpy as np

def defuzzify(fuzzy_y, target_centroids):
	'''
	Inputs:
		fuzzy_y: fuzzy output as [consequent set of rule, firing strength]
		target_centroids: list of all the target centroids (sorted)

	Output: vector of crisp output
	'''
	numerator = 0
	denominator = 0
	for fy in fuzzy_y:
		# firing strength * centroid
		numerator += (fy[1] * target_centroids[fy[0]-1])
		# divided by firing strength
		denominator += fy[1]
	# if no rules are fired, take the mean of the output sets
	if denominator == 0:
		crisp = np.mean(target_centroids)
	else:
		crisp = float(numerator)/float(denominator)
	return(crisp)
