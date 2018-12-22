# SET 7:
# We will compare PLA to SVM with hard margin on linearly separable data sets.
# NB: this program needs tweaking and does not give correct results.

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import math
from random import random
import matplotlib.pyplot as plt

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

def gen_data(N):
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
	# print(m1, b1)

	# collection of N randomly generated points and their outputs
	mis_points = []		# list of misclassified points
	y_points = []
	while True:
		for i in range(N):
			x = random()*2-1
			y = random()*2-1
			mis_points.append([[1.0, x, y], getOutput(x, y, m1, b1)])
			y_points.append(getOutput(x, y, m1, b1))
		if (-1 in y_points) and (1 in y_points):
			break
	mis_length = N 	# length of this list
	return m1, b1, mis_points, mis_length


def PLA_SVM(N):
	# PLA
	m1, b1, mis_points, mis_length = gen_data(N)
	points = mis_points.copy()
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
	# return [iterations, count / float(times)]


	X_new = []
	Y = []
	for p in points:
		X_new.append([p[0][1], p[0][2]])
		Y.append(p[1])
	l = len(Y)

	P = [[] for n in range(l)]
	for r in range(l):
		for c in range(l):
			X_t = np.matrix(X_new[r])
			X = np.matrix.transpose(np.matrix(X_new[c]))
			P[r].append(np.asscalar(Y[r]*Y[c]*np.matmul(X_t, X)))

	q = [-1.0 for n in range(l)]
	G = [[] for n in range(l)]
	for i in range(l):
		for j in range(i):
			G[i].append(0)
		G[i].append(-1)
		for j in range(i+1, l):
			G[i].append(0)
	h = [0.0 for i in range(l)]
	A = np.matrix(Y)

	P = matrix(np.array(P), tc='d')
	q = matrix(q, tc='d')
	G = matrix(G, tc='d')
	h = matrix(h, tc='d')
	A = matrix(A, tc='d')
	b = matrix(0.0)
	solvers.options['show_progress'] = False
	sol = solvers.qp(P, q, G, h, A, b)
	alpha =	sol['x']
	
	w = np.matrix([0.0, 0.0])
	support_vec = []
	for i in range(l):
		if alpha[i] > 10**-4:
			w += (alpha[i] * Y[i] * np.matrix(X_new[i]))
			support_vec.append(i)
	w = w.tolist()
	weight2 = [w[0][0], w[0][1]]
	i = support_vec[0]
	b = 1.0 / Y[i] - (weight2[0]*X_new[i][0] + weight2[1]*X_new[i][1])
	weight2.insert(0, b)

	# calculate P(f =/= g):
	count2 = 0
	times2 = 100
	for i in range(times):
		x = random()*2-1
		y = random()*2-1
		if (getSign(weight2[0]*1 + weight2[1]*x + weight2[1]*y) \
			!= getOutput(x, y, m1, b1)):
			count2 += 1.0
	if (count2 / float(times2)) < (count / float(times)):
		return 1, len(support_vec)
	else:
		return 0, len(support_vec)



def main():
	N = 10
	runs = 1000
	total_times = 0
	ave_support = 0
	for i in range(runs):
		t, a = PLA_SVM(N)
		total_times += t
		ave_support += a
	print(float(total_times) / runs)
	print(float(ave_support) / runs)

	# Uncomment for N = 100 data
	# N = 100
	# runs = 1000
	# total_times = 0
	# ave_support = 0
	# for i in range(runs):
	# 	print(i)
	# 	t, a = PLA_SVM(N)
	# 	total_times += t
	# 	ave_support += a
	# print(float(total_times) / runs)
	# print(float(ave_support) / runs)


if __name__ == '__main__':
	main()



