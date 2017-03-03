from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import csv
import matplotlib.pyplot as plt
import membership_function
import itertools

def cluster(data, target_col, centroid_list, plot=True):
	'''
	Input: numpy array: nummpy array of size > 2x2
	Target_col: integer index of the target column
	Centroid: either a integer (for each feature te same)
	          or an array size = number of features
	plot = Flag, True if you want to see the centroids and 
			clusters (for tuning), flase if not

	Output: centroids of the clusters
	'''
	# check input
	try:
		shape = np.shape(data)
		if(shape[0] <2 or shape[1]<2):
			return -1
	except:
		print("Data should be a numpy aray with minimum size of 2x2")
	feature_count = np.shape(data)[1]
	# TODO: check rest of input
	#!!!!!!!!!!!!!!!!!!!!!!!!!!#
	# get for each feature a centroid count
	if str(centroid_list).isdigit():
		centroid_list = np.full(feature_count, centroid_list, dtype=np.int)
	elif len(centroid_list) != feature_count:
		print("Non valid input for centroid: should be int or array of size number of features")
		return(-1)
	collected = []
	# get target vector 
	Y = data[:, target_col]
	for i in range(feature_count):
		# get number of centroids
		centroid_count = centroid_list[i]
		# get X vector
		# X = np.around(data[:, i], decimals=2)
		X = data[:, i]

		# get the 'for each x the mean y' list
		stack = np.column_stack((X,Y))
		mean_dict = {}
		#for each x collect y's
		mean_list = []
		for xy in stack:
		# for each x collect corresponding mean y
			mean_dict.setdefault(xy[0], []).append(xy[1])
		for x in mean_dict:
			mean_list.append([x, np.mean(mean_dict[x])])
		mean_data = np.array(mean_list)
		# perform kmeans
		kmeans = KMeans(n_clusters=centroid_count, random_state=0).fit(mean_data)
		# plot to see f satisfied and return clusters
		centroids = kmeans.cluster_centers_
		collected.append(centroids)
		labels = kmeans.labels_
		if plot:
			plt.title(" Clusters and centroids for feature " + str(i))
			plt.scatter(mean_data[:,0], mean_data[:,1], c = labels,s=50)
			cx = [float(xy[0]) for xy in centroids]
			cy = [float(xy[1]) for xy in centroids]
			cplot = plt.scatter(cx,cy, marker="o", s=300, color='r', label = 'Cluster centroid',alpha=0.5)
			#plt.rcParams['legend.numpoints'] = 1
			plt.legend(handles=[cplot], numpoints = 1 )
			plt.show()

	# we only use the x coordinates
	x_centroids = [c[c[:,0].argsort()][:, 0] for c in collected]
	return(x_centroids)


def fuzzify(x, centroids, overlap, mf):
	'''
	Inputs: 
		x = nummpy vector of size > 1
		centroids = numpy array with for each feature,
					the centroids for the fuzzy sets
					Size should be len(x) * number of sets
		overlap: number between 0 and 1, 
					when gaussian mf overlap is variance
					when triangle/trapezoid overlap is half of the base
		mf: 'Gaussian', 'triangle', or 'trapezoid' 

	Output: a fizzified datapoint: gor each feature, for each set the 
			membership in degrees
	'''
	# for every feature of the data point
	mship_all_features = []
	for i in range(len(x)):
		feature_value = x[i]
		# for each feature, get centroids of the fuzzy sets
		# sort sets based on x axis
		c = centroids[i]
		# for each fuzzy set centroid
		# depending in membership function, get membership
		mships_per_set = []
		for fset in range(len(c)):
			set_c = c[fset]
			if mf == 'Gaussian':
				mship = membership_function.Gaussian([set_c, overlap], feature_value)
			elif mf == 'triangle':
				mship = membership_function.triangle([set_c-overlap, set_c, set_c+overlap], feature_value)
			elif mf == 'trapezoid':
				mship = membership_function.trapezoid([set_c-overlap, set_c-(overlap/2), set_c+(overlap/2) ,set_c+overlap], feature_value)
			else:
				print(mf, ' is a non valid membership function, please use Gaussian, triangle or trapezoid')
			# only use when not zero
			mships_per_set.append(mship)
		# append to list, that holds all fired sets for all features
		mship_all_features.append(mships_per_set)
	return mship_all_features

def scale(data):
	'''
	Scales data between 0 and 1

	Input: numpy array with rows = data points and
			columns are the features
	Output: scaled numpy array and max and min values 
	'''
	scaled_data = []
	max_x = np.max(data, 0)
	min_x = np.min(data, 0)
	diff = np.subtract(max_x, min_x)
	# xscaled = (x-xmin)/(xmax-xmin)
	# x = xs(xmax-xmin) + xmin
	for x in data:
		scaled_x = np.divide(np.subtract(x, min_x), diff)
		scaled_data.append(scaled_x)
	return(np.array(scaled_data), min_x, max_x)
	

