from math import fabs, log, tan, exp
import numpy as np

def getSign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

def get_in_matrices():
	f = open("in.txt", "r")
	X = []
	Y = []
	for line in f:
		add = line.split()
		X.append([float(add[0]), float(add[1])])
		Y.append(float(add[2]))
	f.close()

	# Q1, 2
	train_X = X[:25]
	train_Y = Y[:25]
	val_X = X[25:]
	val_Y = Y[25:]
	
	## Uncomment below for Q3, 4
	# val_X = X[:25]
	# val_Y = Y[:25]
	# train_X = X[25:]
	# train_Y = Y[25:]

	return train_X, train_Y, val_X, val_Y

def get_out_matrices():
	# out of sample points:
	g = open('out.txt', 'r')
	X_out = []
	Y_out = []
	for line in g:
		add = line.split()
		X_out.append([float(add[0]), float(add[1])])
		Y_out.append(float(add[2]))
	g.close()

	return X_out, Y_out

def class_error(k):
	train_X, train_Y, val_X, val_Y = get_in_matrices()
	trans_X = []
	for x in train_X:
		trans = [1.0, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], \
		         fabs(x[0]-x[1]), fabs(x[0]+x[1])]
		trans_X.append(trans[:k+1])
	X_matrix = np.matrix(trans_X)
	y_matrix = np.matrix.transpose(np.matrix(train_Y))
	# linear regression
	pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
	w = np.matmul(pseudo_inv, y_matrix)

	# transform validation set to Z space
	trans_X_val = []
	for x in val_X:
		trans = [1.0, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], \
		         fabs(x[0]-x[1]), fabs(x[0]+x[1])]
		trans_X_val.append(trans[:k+1])

	# calculate classification error on validation set
	count1 = 0
	for i in range(len(val_X)):
		x = trans_X_val[i]
		y = int(val_Y[i])
		total = 0
		for j in range(k+1):
			total += w[j]*x[j]
		if (getSign(total) != y):
			count1 += 1
	# return float(count1) / len(trans_X_val)

	X_out, Y_out = get_out_matrices()
	trans_X_out = []
	for x in X_out:
		trans = [1.0, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], \
		         fabs(x[0]-x[1]), fabs(x[0]+x[1])]
		trans_X_out.append(trans[:k+1])

	# calculate out-of-sample classification error:
	count2 = 0
	for i in range(len(trans_X_out)):
		x = trans_X_out[i]
		y = int(Y_out[i])
		total = 0
		for j in range(k+1):
			total += w[j]*x[j]
		if (getSign(total) != y):
			count2 += 1
	return float(count1) / len(trans_X_val), float(count2) / len(trans_X_out)



print("k=3:", class_error(3))
print("k=4:", class_error(4))
print("k=5:", class_error(5))
print("k=6:", class_error(6))
print("k=7:", class_error(7))











