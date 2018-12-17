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

def linRegWithoutTransform():
	# generate N = 1000 points on X = [-1,1] x [-1,1]
	x_points = []
	f_points = []
	for i in range(1000):
		x1 = random()*2-1
		x2 = random()*2-1
		f = getSign(x1*x1 + x2*x2 - 0.6)

		x_points.append([1.0, x1, x2])
		f_points.append(f)

	# simulate noise
	flips = np.random.choice(1000, size=100, replace=False)
	for i in flips:
		f_points[i] *= -1.0

	X_matrix = np.matrix(x_points)
	y_matrix = np.matrix.transpose(np.matrix(f_points))

	pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
	w = np.matmul(pseudo_inv, y_matrix)

	i = 0
	count = 0
	for x in x_points:
		if getSign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]) != f_points[i]:
			count += 1
		i += 1

	return float(count) / 1000.0

def linRegWithTrans():
	# generate N = 1000 points on X = [-1,1] x [-1,1]
	x_points = []
	f_points = []
	for i in range(1000):
		x1 = random()*2-1
		x2 = random()*2-1
		f = getSign(x1*x1 + x2*x2 - 0.6)

		x_points.append([1.0, x1, x2, x1*x2, x1*x1, x2*x2])
		f_points.append(f)

	# simulate noise
	flips = np.random.choice(1000, size=100, replace=False)
	for i in flips:
		f_points[i] *= -1.0

	X_matrix = np.matrix(x_points)
	y_matrix = np.matrix.transpose(np.matrix(f_points))

	pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
	w = np.matmul(pseudo_inv, y_matrix)

	# generate N = 1000 points on X = [-1,1] x [-1,1]
	x_points = []
	f_points = []
	for i in range(1000):
		x1 = random()*2-1
		x2 = random()*2-1
		f = getSign(x1*x1 + x2*x2 - 0.6)

		x_points.append([1.0, x1, x2, x1*x2, x1*x1, x2*x2])
		f_points.append(f)

	# simulate noise
	flips = np.random.choice(1000, size=100, replace=False)
	for i in flips:
		f_points[i] *= -1.0

	count = 0
	for i in range(1000):
		if getSign(w[0]*x_points[i][0]+w[1]*x_points[i][1]+w[2]*x_points[i][2] \
				   +w[3]*x_points[i][3]+w[4]*x_points[i][4]+w[5]*x_points[i][5]) \
			!= f_points[i]:
			count += 1
	return w, float(count) / 1000.0


def main():
	s = 0.0
	for i in range(1000):
		s += linRegWithoutTransform()
	print(s / 1000.0)

	weights = [0.0 for i in range(6)]
	result = linRegWithTrans()
	for i in range(10000):
		weights[0] += result[0][0]
		weights[1] += result[0][1]
		weights[2] += result[0][2]
		weights[3] += result[0][3]
		weights[4] += result[0][4]
		weights[5] += result[0][5]
	for i in range(6):
		weights[i] = weights[i] / 10000.0
	print("weights:", weights)
	print("E_out:", result[1])


if __name__ == '__main__':
	main()