
#########################################################
# Training a FIS and testing with a FIS
#
# Functions: train, write, read and test
#########################################################

from Fuzzification import scale, fuzzify, cluster
import membership_function
import WM
import COR
import csv
import numpy as np
from Inference import infer
from error_measures import RMSE, MAE

def read(file):
	'''
	reads a FIS file written by the FIS.write function
	and returns all parameters needed voor testing

	Inputs: 
		file: patch to FIS file

	Outputs:
		method: string
		mf: traingle, Gaussian or trapezoid
		overlap: 0-1 number
		target_centroids: list if float centroids of normalized
						 training target data
		feature_centroids: list if float lists with centroids of 
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

def train(data, target_col, mf, Ncentroids, overlap, method = 'WM', iterations = 1):
	'''
	trains a fis, returns the rules and centroids, needed to write a FIS file
	see FIS.write function

	Inputs: 
		data: nummpy array of size > number of centroids x 2
		target_col: integer index of the target column
		Ncentroids: either a integer (for each feature te same)
	          or an array size = number of features
		mf:  'triangle', 'trapezoid' or 'Gaussian'
		overlap: number between 0 and 1, 
					when gaussian mf overlap is variance
					when triangle/trapezoid overlap is half of the base 
		method: 'WM' or 'COR'
		iterations: number of iterations for the COR algorithm

	Outputs:
		RB: list of lists if integer rules
		target_centroids: list with scaled target centroids
		feature_centroids: the other feature centroids
	'''
	# scale the data
	data, min_x, max_x = scale(data)
	# get centroids 
	centroids = cluster(data, target_col, Ncentroids, plot=False)
	# seperate the target centroids
	if method == 'WM':
		# learn WM rules
		RB = WM.learn(data, centroids, overlap, mf, target_col)
	elif method == 'COR':
		RB = COR.learn(data, centroids, overlap, mf, target_col, min_x[0], max_x[0], iterations)

	# return everything needed for testing
	target_centroids = centroids[target_col]
	feature_centroids = np.delete(centroids,target_col, 0)
	return(RB, target_centroids, feature_centroids)

def test(data, mf, overlap, target_centroids, feature_centroids, RB):
	'''

	Tests a FIS using the parameters of a FIS and test data

	Inputs: 
		data: nummpy array of size > number of centroids x 2
		target_col: integer index of the target column
		target_centroids: list with scaled target centroids
		feature_centroids: the other feature centroids
		
	Outputs:
		RMSE: rooth mean squared error
		MAE: mean absolute error
	'''
	target_col = 0
	data, min_x, max_x = scale(data)
	# remove targets
	targets = data[:, target_col]
	data = np.delete(data, target_col, 1)
	# get crisp output

	crisp = infer(data, RB, feature_centroids, overlap, mf, target_centroids)

	# get errors
	rmse = RMSE(min_x[target_col], max_x[target_col], crisp, targets)
	mae = MAE(min_x[target_col], max_x[target_col], crisp, targets)
	print(rmse)
	print(mae)

