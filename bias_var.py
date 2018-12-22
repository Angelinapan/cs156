# SET 4:
# Exploring the bias-variance model. Assume that training set has only two
# examples (independent) and that the learning algorithm produces the
# hypothesis that minimizes the mean squared error on the examples.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import random

# Target function
def f(x):
	return np.sin(np.pi * x)


# for h(x) = ax:
def calcExpectedValue(N):
	total = 0.0
	g_list = []
	for i in range(N):
		# Generate two random points in [-1, 1] x [-1, 1]
		x1 = random()*2-1
		x2 = random()*2-1
		y1 = f(x1)
		y2 = f(x2)
		X_vector = [x1, x2]
		y_vector = [y1, y2]
		X_matrix = np.matrix.transpose(np.matrix(X_vector))
		y_matrix = np.matrix.transpose(np.matrix(y_vector))
		pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
		a = np.matmul(pseudo_inv, y_matrix)

		g_list.append(a)
		total += a
	return total / float(N), g_list

# for h(x) = ax:
def calcBias(N):
	a = calcExpectedValue(100000)[0]
	total = 0.0
	for i in range(N):
		x1 = random()*2-1
		y1 = f(x1)
		temp = (h(x1, a) - y1)**2
		total += temp
	return total / float(N)

# for h(x) = ax:
def calcVariance(N):
	a, g = calcExpectedValue(50000)
	total = 0.0
	for i in range(N):
		x1 = random()*2-1
		y1 = f(x1)
		exp_total = 0.0
		for j in g:
			exp_total += (h(x1, j) - h(x1, a))**2
		total += exp_total / float(len(g))
	return total / float(N)


def calcExpectedValue_A(N):
	total = 0.0
	g_list = []
	for i in range(N):
		# Generate two random points in [-1, 1] x [-1. 1]
		x1 = random()*2-1
		x2 = random()*2-1
		y1 = f(x1)
		y2 = f(x2)
		b = 0.5 * (y1 + y2)

		g_list.append(b)
		total += b
	return total / float(N), g_list

# h(x) = b:
def E_out_error_A(N):
	b, g = calcExpectedValue_A(10000)
	total_bias = 0.0
	total_var = 0.0
	for i in range(N):
		x1 = random()*2-1
		y1 = f(x1)
		temp = (b - y1)**2
		total_bias += temp
		exp_total = 0.0
		for j in g:
			exp_total += (j - b)**2
		total_var += exp_total / float(len(g))
	
	return (total_bias + total_var) / float(N)

	
# h(x) = ax + b:
def calcExpectedValue_C(N):
	total_a = 0.0
	total_b = 0.0
	a_list = []
	b_list = []
	for i in range(N):
		# Generate two random points in [-1, 1] x [-1. 1]
		x1 = random()*2-1
		x2 = random()*2-1
		y1 = f(x1)
		y2 = f(x2)
		X_vector = [[x1, 1], [x2, 1]]
		y_vector = [y1, y2]
		X_matrix = np.matrix(X_vector)
		y_matrix = np.matrix.transpose(np.matrix(y_vector))
		pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
		a, b = np.matmul(pseudo_inv, y_matrix)

		a_list.append(a)
		b_list.append(b)
		total_a += a
		total_b += b

	return total_a / float(N), total_b / float(N), [a_list, b_list]

def E_out_error_C(N):
	a, b, g = calcExpectedValue_C(1000)
	total_bias = 0.0
	total_var = 0.0
	for i in range(N):
		x1 = random()*2-1
		y1 = f(x1)
		temp = (a*x1+b - y1)**2
		total_bias += temp
		exp_total = 0.0
		for j in range(len(g[0])):
			exp_total += (g[0][j]*x1+g[1][j] - (a*x1+b))**2
		var = exp_total / float(len(g[0]))
		total_var += var
	return (total_bias + total_var) / float(N)


# h(x) = ax^2:
def calcExpectedValue_D(N):
	total = 0.0
	g_list = []
	for i in range(N):
		# Generate two random points in [-1, 1] x [-1. 1]
		x1 = random()*2-1
		x2 = random()*2-1
		y1 = f(x1)
		y2 = f(x2)
		X_vector = [[x1**2], [x2**2]]
		y_vector = [y1, y2]
		X_matrix = np.matrix(X_vector)
		y_matrix = np.matrix.transpose(np.matrix(y_vector))
		pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
		a = np.matmul(pseudo_inv, y_matrix)

		g_list.append(a)
		total += a
	return total / float(N), g_list

def E_out_error_D(N):
	a, g = calcExpectedValue_D(5000)
	total_bias = 0.0
	total_var = 0.0
	for i in range(N):
		x1 = random()*2-1
		y1 = f(x1)
		temp = (a*x1**2 - y1)**2
		total_bias += temp
		exp_total = 0.0
		for j in g:
			exp_total += (j*x1**2 - (a*x1**2))**2
		total_var += exp_total / float(len(g))
	
	return (total_bias + total_var) / float(N)


# h(x) = ax^2 + b:
def calcExpectedValue_E(N):
	total_a = 0.0
	total_b = 0.0
	a_list = []
	b_list = []
	for i in range(N):
		# Generate two random points in [-1, 1] x [-1. 1]
		x1 = random()*2-1
		x2 = random()*2-1
		y1 = f(x1)
		y2 = f(x2)
		X_vector = [[x1**2, 1], [x2**2, 1]]
		y_vector = [y1, y2]
		X_matrix = np.matrix(X_vector)
		y_matrix = np.matrix.transpose(np.matrix(y_vector))
		pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
		a, b = np.matmul(pseudo_inv, y_matrix)

		a_list.append(a)
		b_list.append(b)
		total_a += a
		total_b += b

	return total_a / float(N), total_b / float(N), [a_list, b_list]

def E_out_error_E(N):
	a, b, g = calcExpectedValue_E(1000)
	total_bias = 0.0
	total_var = 0.0
	for i in range(N):
		x1 = random()*2-1
		y1 = f(x1)
		temp = (a*x1**2+b - y1)**2
		total_bias += temp
		exp_total = 0.0
		for j in range(len(g[0])):
			exp_total += (g[0][j]*x1**2+g[1][j] - (a*x1**2+b))**2
		var = exp_total / float(len(g[0]))
		total_var += var
	return (total_bias + total_var) / float(N)


def main():
	# Question 4, 5, 6:
	print(calcExpectedValue(100000)[0])
	print(calcBias(1000))
	print(calcVariance(100))
	print("")
	
	# h(x) = b:
	print(E_out_error_A(1000))
	print("")
	
	# h(x) = ax + b:
	print(E_out_error_C(1000))
	print("")

	# h(x) = ax^2:
	print(E_out_error_D(1000))
	print("")

	# h(x) = ax^2 + b:
	print(E_out_error_E(1000))
	print("")

if __name__ == "__main__":
	main()