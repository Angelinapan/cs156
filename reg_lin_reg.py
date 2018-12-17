from math import fabs, log, tan, exp
import numpy as np

def getSign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

def trans_data(X):
	trans_X = []
	for x in X:
		trans_X.append([1.0, x[1], x[2], x[1]*x[2], x[1]**2, x[2]**2])
	return trans_X

def get_train_matrices(one_digit, trans):
	f = open("train.txt", "r")
	X = []
	Y = []
	for line in f:
		add = line.split()
		X.append([1.0, float(add[1]), float(add[2])])
		if int(add[0][0]) == one_digit:
			Y.append(1.0)
		else:
			Y.append(-1.0)
	f.close()
	trans_X = []
	if trans:
		trans_X = trans_data(X)
	else:
		trans_X = X.copy()
	X_matrix = np.matrix(trans_X)
	y_matrix = np.matrix.transpose(np.matrix(Y))
	return X_matrix, y_matrix, trans_X, Y

def get_test_matrices(one_digit, trans):
	f = open("test.txt", "r")
	X = []
	Y = []
	for line in f:
		add = line.split()
		X.append([1.0, float(add[1]), float(add[2])])
		if int(add[0][0]) == one_digit:
			Y.append(1.0)
		else:
			Y.append(-1.0)
	f.close()
	trans_X = []
	if trans:
		trans_X = trans_data(X)
	else:
		trans_X = X.copy()
	return trans_X, Y

def get_train_1v1(one, two, trans):
	f = open("train.txt", "r")
	X = []
	Y = []
	for line in f:
		add = line.split()
		if int(add[0][0]) == one:
			X.append([1.0, float(add[1]), float(add[2])])
			Y.append(1.0)
		if int(add[0][0]) == two:
			X.append([1.0, float(add[1]), float(add[2])])
			Y.append(-1.0)
	f.close()
	trans_X = []
	if trans:
		trans_X = trans_data(X)
	else:
		trans_X = X.copy()
	X_matrix = np.matrix(trans_X)
	y_matrix = np.matrix.transpose(np.matrix(Y))
	return X_matrix, y_matrix, trans_X, Y

def get_test_1v1(one, two, trans):
	f = open("test.txt", "r")
	X = []
	Y = []
	for line in f:
		add = line.split()
		if int(add[0][0]) == one:
			X.append([1.0, float(add[1]), float(add[2])])
			Y.append(1.0)
		if int(add[0][0]) == two:
			X.append([1.0, float(add[1]), float(add[2])])
			Y.append(-1.0)
	f.close()
	trans_X = []
	if trans:
		trans_X = trans_data(X)
	else:
		trans_X = X.copy()
	return trans_X, Y


def weight_decay_1va(l, one_digit, trans):
	X_matrix, y_matrix, trans_X, Y = get_train_matrices(one_digit, trans)
	pseudo_inv2 = np.matmul(np.linalg.inv(
		                        np.add(np.matmul(np.matrix.transpose(X_matrix), X_matrix),
		                        	   l * np.identity(len(trans_X[0])))), 
		                   np.matrix.transpose(X_matrix))
	w2 = np.matmul(pseudo_inv2, y_matrix)

	# calculate E_in:
	count1 = 0
	for i in range(len(trans_X)):
		x = trans_X[i]
		y = int(Y[i])
		if trans:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]+w2[3]*x[3]+w2[4]*x[4]+w2[5]*x[5]) != y):
				count1 += 1
		else:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]) != y):
				count1 += 1

	# calculate E_out:
	trans_X_out, Y_out = get_test_matrices(one_digit, trans)
	count2 = 0
	for i in range(len(trans_X_out)):
		x = trans_X_out[i]
		y = int(Y_out[i])
		if trans:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]+w2[3]*x[3]+w2[4]*x[4]+w2[5]*x[5]) 
				!= y):
				count2 += 1
		else:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]) != y):
				count2 += 1
	return float(count1) / len(trans_X), float(count2) / len(trans_X_out)


def weight_decay_1v1(l, one, two, trans):
	X_matrix, y_matrix, trans_X, Y = get_train_1v1(one, two, trans)
	pseudo_inv2 = np.matmul(np.linalg.inv(
		                        np.add(np.matmul(np.matrix.transpose(X_matrix), X_matrix),
		                        	   l * np.identity(len(trans_X[0])))), 
		                   np.matrix.transpose(X_matrix))
	w2 = np.matmul(pseudo_inv2, y_matrix)

	# calculate E_in:
	count1 = 0
	for i in range(len(trans_X)):
		x = trans_X[i]
		y = int(Y[i])
		if trans:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]+w2[3]*x[3]+w2[4]*x[4]+w2[5]*x[5]) != y):
				count1 += 1
		else:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]) != y):
				count1 += 1

	# calculate E_out:
	trans_X_out, Y_out = get_test_1v1(one, two, trans)
	count2 = 0
	for i in range(len(trans_X_out)):
		x = trans_X_out[i]
		y = int(Y_out[i])
		if trans:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]+w2[3]*x[3]+w2[4]*x[4]+w2[5]*x[5]) 
				!= y):
				count2 += 1
		else:
			if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]) != y):
				count2 += 1
	return float(count1) / len(trans_X), float(count2) / len(trans_X_out)

print(weight_decay(1, 0, True))
print(weight_decay(1, 1, True))
print(weight_decay(1, 2, True))
print(weight_decay(1, 3, True))
print(weight_decay(1, 4, True))

print(weight_decay_1v1(0.01, 1, 5, True))
print(weight_decay_1v1(1, 1, 5, True))

print("Transformed:")
print("0:", weight_decay_1va(1, 0, True))
print("1:", weight_decay_1va(1, 1, True))
print("2:", weight_decay_1va(1, 2, True))
print("3:", weight_decay_1va(1, 3, True))
print("4:", weight_decay_1va(1, 4, True))
print("5:", weight_decay_1va(1, 5, True))
print("6:", weight_decay_1va(1, 6, True))
print("7:", weight_decay_1va(1, 7, True))
print("8:", weight_decay_1va(1, 8, True))
print("9:", weight_decay_1va(1, 9, True))

print("Not Transformed:")
print("0:", weight_decay_1va(1, 0, False))
print("1:", weight_decay_1va(1, 1, False))
print("2:", weight_decay_1va(1, 2, False))
print("3:", weight_decay_1va(1, 3, False))
print("4:", weight_decay_1va(1, 4, False))
print("5:", weight_decay_1va(1, 5, False))
print("6:", weight_decay_1va(1, 6, False))
print("7:", weight_decay_1va(1, 7, False))
print("8:", weight_decay_1va(1, 8, False))
print("9:", weight_decay_1va(1, 9, False))















