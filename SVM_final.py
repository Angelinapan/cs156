# Final Exam:
# This program applies the hard-margin SVM algorithm with the 
# (second-order polynomial) kernel:
# 			K(x, x') = (1 + x^T x')^2
# using the sklearn.svm package.

import numpy as np
X = np.array([[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]])
Y = np.array([-1, -1, -1, 1, 1, 1, 1])
from sklearn.svm import SVC
clf = SVC(C=10**6, kernel='poly', degree=2, gamma='auto', coef0=1)
print(clf.fit(X,Y))
print(clf.support_vectors_)