# SET 1:
# This program explores how the Perceptron Learning Algorithm works.

import math
from random import random

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

# Implements the Perceptron Learning Algorithm for d = 2 on a linear data set.
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
	mis_points = []		# list of misclassified points
	for i in range(N):
		x = random()*2-1
		y = random()*2-1
		mis_points.append([[1.0, x, y], getOutput(x, y, m1, b1)])
	mis_length = N 	# length of this list

	weight = [0.0, 0.0, 0.0]
	class_points = []	# list of properly classified points
	class_length = 0	# length of this list
	iterations = 0

	while mis_length > 0: 	# while there are still misclassified points
		i = int(random()*mis_length)
		weight = [weight[0] + mis_points[i][1]*mis_points[i][0][0], \
				  weight[1] + mis_points[i][1]*mis_points[i][0][1], \
				  weight[2] + mis_points[i][1]*mis_points[i][0][2]]
		class_points.append(mis_points[i])
		class_length += 1
		mis_points.pop(i)
		mis_length -= 1
		index = 0
		# update list of properly classified points
		while index < class_length:
			if (getSign(weight[0]*class_points[index][0][0] \
				+ weight[1]*class_points[index][0][1] \
				+ weight[2]*class_points[index][0][2]) \
				!= getSign(class_points[index][1])):
				mis_points.append(class_points[index])
				mis_length += 1
				class_points.pop(index)
				class_length -=1
			else:
				index += 1
		iterations += 1

	# calculate P(f =/= g):
	count = 0
	times = 100
	for i in range(times):
		x = random()*2-1
		y = random()*2-1
		if (getSign(weight[0]*1 + weight[1]*x + weight[1]*y) \
			!= getOutput(x, y, m1, b1)):
			count += 1.0
	return [iterations, count / float(times)]

def calcAverage(iterations, N, value):
	total = 0.0
	for i in range(iterations):
		total += PLA(N)[value]
	return 1.0 * total / iterations

def main():
	# for N = 10:
	print("N = 10:")
	print("iterations: ", calcAverage(1000, 10, 0))
	print("probability: ", calcAverage(1000, 10, 1))

	# for N = 100:
	print("N = 100:")
	print("iterations: ", calcAverage(1000, 100, 0))
	print("probability: ", calcAverage(1000, 100, 1))

if __name__ == "__main__":
	main()











