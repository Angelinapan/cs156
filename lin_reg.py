# SET 2:
# This program explores how Linear Regression for classification works,
# using an input set of X = [-1, 1] x [-1, 1]. 

import math
from random import random
import numpy as np

def getSign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

def getOutput(x, y, m, b):
	if y > m*x+b:
		return 1
	else:
		return -1

def getF(x1, y1, x2, y2):
	m = (y1 - y2) / (x1 - x2)
	b = (x1*y2 - x2*y1) / (x1-x2)
	return [m, b]

# Use linear regression on input set [-1, 1] x [-1. 1] and target function f.
def LinReg(N):
	# Generate two random points in [-1, 1] x [-1. 1]
	x1 = random()*2-1
	y1 = random()*2-1
	x2 = random()*2-1
	y2 = random()*2-1
	f1 = [x1, y1]
	f2 = [x2, y2]
	nums = getF(x1, y1, x2, y2)
	m1 = nums[0]
	b1 = nums[1]

	# collection of N randomly generated points and their outputs
	X_vector = []
	y_vector = []
	for i in range(N):
		x = random()*2-1
		y = random()*2-1
		X_vector.append([1.0, x, y])
		y_vector.append(getOutput(x, y, m1, b1))

	X_matrix = np.matrix(X_vector)
	y_matrix = np.matrix.transpose(np.matrix(y_vector))

	pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
	w = np.matmul(pseudo_inv, y_matrix)

	i = 0
	count = 0
	for x in X_vector:
		if (getSign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]) != y_vector[i]):
			count += 1
		i += 1

	count2 = 0
	for i in range(1000):
		x = random()*2-1
		y = random()*2-1
		if (getSign(w[0]*1+w[1]*x+w[2]*y) != getOutput(x, y, m1, b1)):
			count2 += 1
	return (float(count) / 100.0), (float(count2) / 1000.0)


# Uses weights from Linear Regression as the vector of initial weights for
# the Perceptron Learning Algorithm.
def PLA(N):
	# Generate two random points in [-1, 1] x [-1. 1]
	x1 = random()*2-1
	y1 = random()*2-1
	x2 = random()*2-1
	y2 = random()*2-1
	f1 = [x1, y1]
	f2 = [x2, y2]
	nums = getF(x1, y1, x2, y2)
	m1 = nums[0]
	b1 = nums[1]

	# collection of N randomly generated points and their outputs
	X_vector = []
	y_vector = []
	for i in range(N):
		x = random()*2-1
		y = random()*2-1
		X_vector.append([1.0, x, y])
		y_vector.append(getOutput(x, y, m1, b1))

	X_matrix = np.matrix(X_vector)
	y_matrix = np.matrix.transpose(np.matrix(y_vector))

	pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
	w = np.matmul(pseudo_inv, y_matrix)

	i = 0
	misclassified = []	# set of misclassified points
	mis_length = 0
	classified = []
	class_length = 0
	for x in X_vector:
		if (getSign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]) != y_vector[i]):
			misclassified.append([x, y_vector[i]])
			mis_length += 1
		else:
			classified.append([x, y_vector[i]])
			class_length += 1
	iterations = 0
	while mis_length > 0:
		i = int(random()*mis_length)
		w = [w[0] + misclassified[i][1]*misclassified[i][0][0], \
			 w[1] + misclassified[i][1]*misclassified[i][0][1], \
			 w[2] + misclassified[i][1]*misclassified[i][0][2]]
		classified.append(misclassified[i])
		class_length += 1
		misclassified.pop(i)
		mis_length -= 1

		index = 0
		while index < class_length:
			if getSign(w[0]*classified[index][0][0] + w[1]*classified[index][0][1] \
				+ w[2]*classified[index][0][2]) != classified[index][1]:
				misclassified.append(classified[index])
				mis_length += 1
				classified.pop(index)
				class_length -= 1
			else:
				index += 1
		iterations += 1
	return iterations


def main():
	# for Q5 and Q6:
	print("N = 100:")
	s1 = 0.0
	s2 = 0.0
	for i in range(1000):
		values = LinReg(100)
		s1 += values[0]
		s2 += values[1]

	# finds average E_in
	print("E_in:", s1 / 1000.0)
	# finds average E_out
	print("E_out:", s2 / 1000.0)

	# for Q7:
	print("N = 10:")
	s3 = 0
	for i in range(1000):
		s3 += PLA(10)
	# finds average number of iterations that PLA takes to converge
	print("iterations:", float(s3)/1000.0)

if __name__ == "__main__":
	main()











