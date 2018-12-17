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
	trans_X = []
	for x in X:
		trans_X.append([1.0, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], 
			           fabs(x[0]-x[1]), fabs(x[0]+x[1])])
	X_matrix = np.matrix(trans_X)
	y_matrix = np.matrix.transpose(np.matrix(Y))
	return X_matrix, y_matrix, trans_X, Y

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

	# out of sample points in feature space Z:
	trans_X_out = []
	for x in X_out:
		trans_X_out.append([1.0, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], 
			           fabs(x[0]-x[1]), fabs(x[0]+x[1])])
	return trans_X_out, Y_out

def lin_reg():
	X_matrix, y_matrix, trans_X, Y = get_in_matrices()
	pseudo_inv = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X_matrix), X_matrix)), np.matrix.transpose(X_matrix))
	w = np.matmul(pseudo_inv, y_matrix)
	print(w)

	# calculate E_in in feature space Z:
	count1 = 0
	for i in range(len(trans_X)):
		x = trans_X[i]
		y = int(Y[i])
		if (getSign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+w[3]*x[3]+w[4]*x[4]+w[5]*x[5]
			        +w[6]*x[6]+w[7]*x[7]) 
			!= y):
			count1 += 1

	trans_X_out, Y_out = get_out_matrices()
	count2 = 0
	for i in range(len(trans_X_out)):
		x = trans_X_out[i]
		y = int(Y_out[i])
		if (getSign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+w[3]*x[3]+w[4]*x[4]+w[5]*x[5]
			        +w[6]*x[6]+w[7]*x[7]) 
			!= y):
			count2 += 1
	return float(count1) / len(trans_X), float(count2) / len(trans_X_out)


def weight_decay(k):
	X_matrix, y_matrix, trans_X, Y = get_in_matrices()
	l = 10.0**k
	pseudo_inv2 = np.matmul(np.linalg.inv(
		                        np.add(np.matmul(np.matrix.transpose(X_matrix), X_matrix),
		                        	   l * np.identity(len(trans_X[0])))), 
		                   np.matrix.transpose(X_matrix))
	w2 = np.matmul(pseudo_inv2, y_matrix)

	# calculate E_in in feature space Z:
	count1 = 0
	for i in range(len(trans_X)):
		x = trans_X[i]
		y = int(Y[i])
		if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]+w2[3]*x[3]+w2[4]*x[4]+w2[5]*x[5]
			        +w2[6]*x[6]+w2[7]*x[7]) 
			!= y):
			count1 += 1

	trans_X_out, Y_out = get_out_matrices()
	count2 = 0
	for i in range(len(trans_X_out)):
		x = trans_X_out[i]
		y = int(Y_out[i])
		if (getSign(w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]+w2[3]*x[3]+w2[4]*x[4]+w2[5]*x[5]
			        +w2[6]*x[6]+w2[7]*x[7]) 
			!= y):
			count2 += 1
	return float(count1) / len(trans_X), float(count2) / len(trans_X_out)

def main():
	print("Q2:", lin_reg())
	print("Q3:", weight_decay(-3.0))
	print("Q4:", weight_decay(3.0))
	print("Q5 and 6:")
	print("k=2: ", weight_decay(2.0))
	print("k=1: ", weight_decay(1.0))
	print("k=0: ", weight_decay(0.0))
	print("k=-1: ", weight_decay(-1.0))
	print("k=-2: ", weight_decay(-2.0))

if __name__ == "__main__":
	main()























