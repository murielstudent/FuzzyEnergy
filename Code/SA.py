
from Fuzzification import fuzzify
import membership_function
import numpy as np
import itertools
import random
from Inference import infer
from Defuzzification import defuzzify
from error_measures import RMSE
import matplotlib.pyplot as plt

def search(data, targets,  RB, alpha, feature_centroids, overlap, mf, target_centroids, min_y, max_y, plot=False, iterations = 100):
	'''
	Inputs: 
		data: nummpy array of size > number of centroids x 2
		targets: numpy array with for each example the target value
		RB: the initial Rule Base 
		alpha: scaling factor of the rule base, i.e. alhpa = 0.2
				will return a rule base 0.2* the original size
		feature_centroids = the centroids of the input features
		mf:  'triangle', 'trapezoid' or 'Gaussian'
		overlap: number between 0 and 1, 
					when gaussian mf overlap is variance
					when triangle/trapezoid overlap is half of the base 
		target_centroids = the centroids of the target feature
		min_y and min_x: the smallest and largest value of the target
						feature, before scaling, in order to scale back
		plot = boolean, if true then the SA process is plotted
		iterations: how many iterations to run the SA


	Output: integer Rule base = list of size (rules * features)
	'''
	# set fitness at time 0
	cost = 999999

	# perform simulated annealing
	cost_list = []
	iter_list = []
	for i in range(iterations):
		Nrules = int(alpha * len(RB))
		# T will decrease over time for smoothing out climbing behaviour
		T = 0.1/(i+1)
		# sample Nrules randomly
		RB_sample = random.sample(RB, Nrules)
		# start inference 
		crisp, outliers= infer(data, RB_sample, feature_centroids, overlap, mf, target_centroids)
		# get RMSE of this random sample
		cost_new = RMSE(min_y, max_y, crisp, targets)
		# keep the rule base with lowest fitness cost
		if cost_new < cost:
			# climb to next node in search space
			RB_best = RB_sample
			cost = cost_new
		else:
			# get probability of climbing anyways
			climb_prob = np.exp((cost-cost_new)/T)
			# toss coin
			if (random.random() < climb_prob):
				# climb to next node in search space
				RB_best = RB_sample
				cost = cost_new
			# else stay on same node
		cost_list.append(cost)
		iter_list.append(i)
	if plot:
		plt.xlabel('Epoch')
		plt.ylabel('RMSE')
		plt.title('Simulated Annealing Process')
		plt.plot(iter_list, cost_list)
		plt.show()
	return(RB_best)




	




	
	
