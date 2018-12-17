from math import exp, sqrt, e, log
import numpy as np
from random import random, shuffle

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
	return m, b

def log_reg(N, n):
	# Generate two random points in [-1, 1] x [-1, 1]
	x1 = random()*2-1
	y1 = random()*2-1
	x2 = random()*2-1
	y2 = random()*2-1
	f1 = [x1, y1]
	f2 = [x2, y2]
	m1, b1 = getF(x1, y1, x2, y2)

	# collection of N randomly generated points and their outputs
	x_points = []
	y_points = []
	for i in range(N):
		x1 = random()*2-1
		x2 = random()*2-1
		x_points.append([[1.0], [x1], [x2]])
		y_points.append(getOutput(x1, x2, m1, b1))

	
	# Initialize weights to be all 0
	W_matrix = np.matrix([[0.0], [0.0], [0.0]])

	epoch = 0
	# SGD loop.
	while True:
		points = [j for j in range(N)]
		shuffle(points)
		temp_w = W_matrix.copy()  # temporary copy of weight vector from previous epoch.
		for n in points:
			temp = y_points[n] * np.matmul(np.matrix.transpose(temp_w), np.matrix(x_points[n]))
			if temp < 700.0:
				grad = -1.0 * y_points[n] * np.matrix(x_points[n]) / (1.0 + exp(temp))
				np.subtract(temp_w, float(n) * grad, out=temp_w)
		epoch += 1
		# Stop SGD algorithm when ||w_(t-1) - w_t|| < 0.01
		if np.linalg.norm(np.subtract(W_matrix, temp_w)) < 0.01:
			break
		W_matrix = temp_w.copy()

	# calculate E_out, the cross entropy error:
	times = 1000
	total = 0.0
	for i in range(times):
		x1 = random()*2-1
		x2 = random()*2-1
		y = getOutput(x1, x2, m1, b1)
		power = -1.0 * y * np.matmul(np.matrix.transpose(W_matrix), np.matrix([[1.0], [x1], [x2]]))
		total += log(1.0 + exp(power))

	print(float(total) / float(times), epoch)
	return float(total) / float(times), epoch

def main():
	total_error = 0.0
	total_epochs = 0.0
	for i in range(100):
		total = log_reg(100, 0.01)
		total_error += total[0]
		total_epochs += total[1]
	print(total_error / 100.0)
	print(total_epochs / 100.0)

if __name__ == '__main__':
	main()







