# Mutlivariate Linear Regression

3 Python files included. A solution using the normal equation, and two solutions using gradient descent.
Of the two gradient descent solutions included, there's an inefficient and more efficient solution as indicated by their file names. 

The multivariate gradient descent solution is able to:
-Read a file where each each column corresponds to a new feature and the rightmost column is the y-values for each training example.
-Perform gradient descent
-Plot a graph of the cost function against the number of iterations, for the algorithm.

The features are contained in X, a n+1 x m  dimensional matrix. Each new column is a different feature, and each row is a different training example. The first element of each row is an added bias unit.
y is a m dimensional vector.
theta is a n+1 dimensional vector.

Normalisation is performed on X to speed up gradient descent.
