from math import sin, pi, exp
from random import random
import numpy as np
from sklearn.svm import SVC

def getSign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

def target(x1, x2):
	value = x2 - x1 + 0.25 * sin(pi * x1)
	if value > 0:
		return 1
	elif value < 0:
		return -1
	else:
		return 0

def dist(X1, X2):
	return (X2[0]-X1[0])**2 + (X2[1]-X1[1])**2

def gen_points(N):
	train_x = []
	train_y = []
	for i in range(N):
		x1 = random()*2 - 1
		x2 = random()*2 - 1
		train_x.append([x1, x2])
		train_y.append(target(x1, x2))
	return np.array(train_x), np.array(train_y)

def kernel_RBF(X, Y, g):
	clf = SVC(C=10**6, kernel='rbf', gamma=g)
	clf.fit(X,Y)
	## Uncomment for Q13 only:
	# if clf.fit_status_ == 1:
	# 	return 1
	# else:
	# 	return 0

	# Calculate E_out:
	count = 0
	for i in range(1000):
		x1 = random()*2 - 1
		x2 = random()*2 - 1
		if clf.predict([[x1, x2]])[0] != target(x1, x2):
			count += 1
	# For E_in and E_out:
	return clf.fit_status_, float(count) / 1000

def reg_RBF(X, Y, g, M, K):
	# Lloyd's Algorithm to find centres:

	# Initialize centres to random points in X
	mu = []
	for i in range(K):
		x1 = random()*2 - 1
		x2 = random()*2 - 1
		mu.append([x1, x2])

	count = 1
	while True:
		# find clusters:
		clusters = [[] for m in mu]
		for x in X:
			least = 10	# least distance
			S = -1		# index of cluster
			for i in range(K):
				d = dist(x, mu[i])
				if d < least:
					least = d
					S = i
			clusters[S].append(x)
		# find mus:
		new_mu = []
		for k in range(K):
			total = np.array([0.0, 0.0])
			for x in clusters[k]:
				total += x
			length = len(clusters[k])
			if length == 0:
				return -1
			else:
				new_mu.append(total / length)
		if np.array_equal(new_mu, mu):
			break
		else:
			mu = new_mu.copy()
		count += 1

	# Calculating Phi matrix:
	Phi = []
	for x in X:
		row = []
		for k in range(K):
			row.append(exp(-g * dist(x, mu[k])))
		row.append(1.0)
		Phi.append(row)
	Phi_matrix = np.matrix(Phi)
	Phi_t = np.matrix.transpose(Phi_matrix)
	y_matrix = np.matrix.transpose(np.matrix(Y))

	# Calculate pseudo-inverse:
	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(Phi_t, Phi_matrix)), Phi_t), y_matrix)

	# Calculate E_in:
	count = 0
	for i in range(len(X)):
		total = 0
		for k in range(K):
			total += w[k] * exp(-g * dist(X[i], mu[k]))
		total += w[-1]
		if getSign(total) != Y[i]:
			count += 1

	# Calculate E_out:
	count2 = 0
	for i in range(1000):
		x1 = random()*2 - 1
		x2 = random()*2 - 1
		x = [x1, x2]
		y = target(x1, x2)
		total2 = 0
		for k in range(K):
			total2 += w[k] * exp(-g * dist(x, mu[k]))
		total2 += w[-1]
		if getSign(total2) != y:
			count2 += 1
	# For E_in and E_out:
	return float(count) / len(X), float(count2) / 1000
	# return float(count2) / 1000

def main():
	# Q13:
	for i in range(1000):
		# print(i)
		X, Y = gen_points(100)
		print(kernel_RBF(X, Y, 1.5))

	# Q14:
	count = 0
	i = 0
	while i < 100:
		X, Y = gen_points(100)
		reg_value = reg_RBF(X, Y, 1.5, 1, 9)
		kern_value = kernel_RBF(X, Y, 1.5)
		print(reg_value, kern_value)
		if reg_value != -1 and kern_value != -1:
			if kern_value[-1] < reg_value[-1]:
				count += 1
		else:
			continue
		i += 1
	print("K=9:", count)

	# Q15:
	count = 0
	i = 0
	while i < 100:
		X, Y = gen_points(100)
		reg_value = reg_RBF(X, Y, 1.5, 1, 12)
		kern_value = kernel_RBF(X, Y, 1.5)
		print(reg_value, kern_value)
		if reg_value != -1 and kern_value != -1:
			if kern_value[-1] < reg_value[-1]:
				count += 1
		else:
			continue
		i += 1
	print("K=12:", count)

	# Q16:
	a_count = 0
	b_count = 0
	c_count = 0
	d_count = 0
	e_count = 0
	for i in range(100):
		X, Y = gen_points(100)
		reg_RBF_9 = reg_RBF(X, Y, 1.5, 1, 9)
		reg_RBF_12 = reg_RBF(X, Y, 1.5, 1, 12)
		if reg_RBF_9 == -1 or reg_RBF_12 == -1:
			continue
		E_in_9, E_out_9 = reg_RBF_9
		E_in_12, E_out_12 = reg_RBF_12
		if E_in_12 - E_in_9 < 0 and E_out_12 - E_out_9 > 0:
			a_count += 1
		elif E_in_12 - E_in_9 > 0 and E_out_12 - E_out_9 < 0:
			b_count += 1
		elif E_in_12 - E_in_9 > 0 and E_out_12 - E_out_9 > 0:
			c_count += 1
		elif E_in_12 - E_in_9 < 0 and E_out_12 - E_out_9 < 0:
			d_count += 1
		else:
			e_count += 1
	print(a_count, b_count, c_count, d_count, e_count)

	# Q17:
	a_count = 0
	b_count = 0
	c_count = 0
	d_count = 0
	e_count = 0
	for i in range(1000):
		print(i)
		X, Y = gen_points(100)
		reg_RBF_1 = reg_RBF(X, Y, 1.5, 1, 9)
		reg_RBF_2 = reg_RBF(X, Y, 2.0, 1, 9)
		if reg_RBF_1 == -1 or reg_RBF_2 == -1:
			continue
		E_in_1, E_out_1 = reg_RBF_1
		E_in_2, E_out_2 = reg_RBF_2
		if E_in_2 - E_in_1 < 0 and E_out_2 - E_out_1 > 0:
			a_count += 1
		elif E_in_2 - E_in_1 > 0 and E_out_2 - E_out_1 < 0:
			b_count += 1
		elif E_in_2 - E_in_1 > 0 and E_out_2 - E_out_1 > 0:
			c_count += 1
		elif E_in_2 - E_in_1 < 0 and E_out_2 - E_out_1 < 0:
			d_count += 1
		elif E_in_2 - E_in_1 == 0 and E_out_2 - E_out_1 == 0:
			e_count += 1
		else:
			continue
	print(a_count, b_count, c_count, d_count, e_count)

	# Q18:
	count = 0
	i = 0
	while i < 1000:
		print(i)
		X, Y = gen_points(100)
		value = reg_RBF(X, Y, 1.5, 1, 9)
		if value != -1:
			if value[0] == 0:
				count += 1
			i += 1
	print(float(count) / 1000)
		

if __name__ == '__main__':
	main()





