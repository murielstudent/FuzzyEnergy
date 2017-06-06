#########################################################
# Rooth Mean Squared Error
# Mean Absolute Error
# both after descaling the output of the system first
#########################################################


import numpy as np

def RMSE(min_y, max_y, yhat, y):
	# first scale output and target back to 
	# original scale, to prevent scale bias
	yhat = descale(yhat, min_y, max_y)
	y = descale(y, min_y, max_y)
	return(np.sqrt(np.mean(np.power(np.subtract(yhat,y),2))))

def MAE(min_y, max_y, yhat, y):
	# first scale output and target back to 
	# original scale, to prevent scale bias
	yhat = descale(yhat, min_y, max_y)
	y = descale(y, min_y, max_y)
	return(np.mean(np.absolute(np.subtract(yhat,y))))


def descale(scaled_y, min_y, max_y):
	'''
	Descaled data back to original scale

	Inputs: 
		y = vector of values
		min_y = minimum value of original data
		max_y = minimum value of original data

	Output: y in original scale
	'''
	diff = np.subtract(max_y ,min_y)
	descaled_y = np.add(np.multiply(scaled_y, diff), min_y)
	# descaled y = scaled_y *(ymax-ymin)+ymin
	# descaled_y = [(y*(diff)+min_y) for y in scaled_y]
	return(descaled_y)

	