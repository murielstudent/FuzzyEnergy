# Implementation of several membership functions
# in order to return the membership of a given 
# crisp value.
# 
# source: http://www.csee.wvu.edu/classes/cpe521/presentations/Membership.pdf
#
# 3 different membership functions
# triangle, trapezoid and Gaussian
# 
# input Gaussian: mean, variance and a crisp value
# input traingle: list of 3 membership function values of 1 specific set and a crisp value
# input traingle: list of 4 membership function values of 1 specific set and a crisp value
#
# returns: membership value


import numpy as np

# calculates the membership value of a Gaussian set
def Gaussian(values, x):
	mean = values[0]
	variance = values[1]
	f = np.exp(-(np.divide(((x-mean)**2), (variance**2))))
	return f


# calculates the membership value of trapezoidal set
def trapezoid(values,x):
	a = values[0]
	b = values[1]
	c = values[2]
	d = values[3]

	if b <= x and x < c:
		f = 1
	elif a <= x and x< b:
		f = (x-a)/(b-a)
	elif c==d and c <= x and x<= d:
		f = 1
	elif c <= x and x<= d:
		f = (d-x)/(d-c)
	else:
		f = 0

	return f

# calculates the membership value of triangular set
def triangle(values,x):
	a = values[0]
	b = values[1]
	c = values[2]

	if x == b:
		f = 1
	elif a <= x and x<= b:
		f = (x-a)/(b-a)
	elif b <= x and x<= c:
		f = (c-x)/(c-b)
	else:
		f = 0

	return f

