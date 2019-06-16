import numpy as np
from numpy.linalg import inv

data = np.loadtxt("data/OneFeature.txt", delimiter=",", unpack=True)  # Output is 1D. Vectors are 2D m x 1 dimension so need to change to 2D using reshape

y = data[-1].reshape((data.shape[1], 1))  # Makes y a vector with each new line different training set

onesVector = np.ones((data.shape[1], 1))  # Creates an m dimensional vector of 1s
X = np.concatenate((onesVector, data[:-1].T), axis=1)  # Creates matrix X, where each new line is a different training example, where x0 on each line is 1

theta = inv((X.T.dot(X))).dot(X.T).dot(y)
print("Theta = ", theta)