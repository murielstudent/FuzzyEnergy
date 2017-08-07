# FIS: Fuzzy Inference System
# Training a FIS and testing with a FIS
#
# Functions: train, write, read and test

import numpy as np
import pickle
import csv

from Fuzzification import scale, fuzzify, cluster
from Inference import infer
from error_measures import RMSE, MAE, descale
import membership_function
import WM
import SA

def read(file):
	'''
	reads a FIS file written by the FIS.write function
	and returns all parameters needed voor testing

	Inputs: 
		file: path to FIS file

	Outputs:
		method: string
		mf: traingle, Gaussian or trapezoid
		overlap: [0,1] float
		target_centroids: list with centroids of normalized
						 training target data
		feature_centroids: list of lists with centroids of 
						normalized training features data 
						
		RB: List of lists holding the integer rules
	'''
	feature_centroids = []
	RB = []
	with open(file) as fis_file:
		reader = fis_file.readlines()
		for i in range(len(reader)):
			if 'Method' in reader[i]:
				i += 1
				method = reader[i].replace("\n", "")
			elif 'membership function' in reader[i]:
				i += 1
				mf = reader[i].replace("\n", "")
			elif 'overlap' in reader[i]:
				i += 1
				overlap = float(reader[i].replace("\n", ""))
			elif 'target centroids' in reader[i]:
				i += 1
				target_centroids = [float(x) for x in reader[i].split()]
			elif 'feature centroids' in reader[i]:
				i += 1
				while True:
					line = reader[i]	
					if line == '\n':
						break	
					feature_centroids.append([float(x) for x in line.split()])
					i += 1
			elif 'Rules' in reader[i]:
				i += 1
				try:
					while True:
						line = reader[i]	
						if len(line)<2:
							break	
						RB.append([int(x) for x in line.split()])
						i += 1
				except:
					IndexError
					
		return method, mf, overlap, target_centroids, feature_centroids, RB

def train(FIS_name, data, target_col, mf, Ncentroids, overlap, alpha = 0.5, iterations = 50, sa = False, sa_plot = False):
	'''
	Trains a FIS, writes all the properties of this FIS to a FIS file
	using the write function.

	Inputs: 
		data: nummpy array of size > number of centroids x 2
		target_col: integer index of the target column
		Ncentroids: either an integer (for each feature te same)
	          or an array size = number of features
		mf:  'triangle', 'trapezoid' or 'Gaussian'
		overlap: number between 0 and 1, 
					when gaussian mf overlap is the variance
					when triangle/trapezoid overlap is half of the base 
		iterations: number of iterations for the simulated annealing

	Outputs:
		RB: list of lists of integer rules
		target_centroids: list with scaled target centroids
		feature_centroids: the other feature centroids
	'''
	# scale the data
	data, min_x, max_x = scale(data)
	# get centroids 
	centroids = cluster(data, target_col, Ncentroids, plot=False)
	# learn WM rules
	RB = WM.learn(data, centroids, overlap, mf, target_col)
	# return everything needed for testing
	target_centroids = centroids[target_col]
	# delete target centroid for testing
	feature_centroids = np.delete(centroids,target_col, 0)
	# delete target values for testing
	targets = data[:, target_col]
	data = np.delete(data,target_col,1)
	method = 'WM'
	# for simulated annealing, get the new rule base
	if sa:
		method = 'WM+SA'
		RB = SA.search(data, targets, RB, alpha, feature_centroids, overlap, mf, target_centroids, min_x[target_col], max_x[target_col], plot = sa_plot, iterations = iterations)
	# Write FIS file in the format:
	# FIS_name.FIS
	with open(FIS_name + '.FIS', "w") as fis_file:
		write(fis_file, method, mf, overlap, target_centroids, feature_centroids, RB)


def test(data, mf, overlap, target_centroids, feature_centroids, RB, target_col, threshold = None):
	'''
	Tests a FIS using the parameters of a FIS and test data

	Inputs: 
		data: nummpy array of size > number of centroids x 2
		target_col: integer index of the target column
		target_centroids: list with scaled target centroids
		feature_centroids: the other feature centroids
		threshold: If None, the mean firing strength is returned. 
					If number, then the outliers are returned
		
	Outputs:
		RMSE: rooth mean squared error
		MAE: mean absolute error
	'''
	data, min_x, max_x = scale(data)

	# remove targets
	targets = data[:, target_col]
	data = np.delete(data, target_col, 1)
	# get crisp output, and output is either a list with all the
	# outliers, or the mean firing strength of the data set
	crisp , output = infer(data, RB, feature_centroids, overlap, mf, target_centroids, th = threshold)
	# get errors
	rmse = RMSE(min_x[target_col], max_x[target_col], crisp, targets)
	mae = MAE(min_x[target_col], max_x[target_col], crisp, targets)

	min_x = np.delete(min_x, 1)
	max_x = np.delete(max_x, 1)

	if threshold:
		return(rmse, mae, len(output))
	else:
		return(rmse, mae, output)

		# TO DO : stich to data base?
		# outlist = []
		# for out in outliers:
		# 	outlist.append(descale(out, min_x, max_x))
		# pickle.dump(outlist, open('outliers_NTH.p', 'wb'))
	
def write(text_file, algorithm, mf, overlap, target_centroids, feature_centroids, RB):
	text_file.write("Method\n")
	text_file.write(algorithm)
	text_file.write('\n')
	text_file.write('\n')
	text_file.write("membership function\n")
	text_file.write(mf)
	text_file.write('\n')
	text_file.write('\n')
	text_file.write("overlap\n")
	text_file.write(str(overlap))
	text_file.write('\n')
	text_file.write('\n')
	text_file.write("target centroids\n")
	for c in target_centroids:
		text_file.write(str(c))
		text_file.write(' ')
	text_file.write('\n')
	text_file.write('\n')
	text_file.write("feature centroids\n")
	for cent in feature_centroids:
		for c in cent:
			text_file.write(str(c))
			text_file.write(' ')
		text_file.write('\n')
	text_file.write('\n')
	text_file.write('Rules\n')
	for rule in RB:
		for r in rule:
			text_file.write(str(r))
			text_file.write(' ')
		text_file.write('\n')
	text_file.close()


