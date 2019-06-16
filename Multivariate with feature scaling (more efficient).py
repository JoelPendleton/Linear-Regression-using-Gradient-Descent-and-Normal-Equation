import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/TwoFeatures.txt", delimiter=",", unpack=True)  # Output is 1D. Vectors are 2D m x 1 dimension so need to change to 2D using reshape

y = data[-1].reshape((data.shape[1], 1))  # Makes y a vector with each new line different training set

theta = np.ones((data.shape[0], 1))

onesVector = np.ones((data.shape[1], 1))  # Creates an m dimensional vector of 1s
X = np.concatenate((onesVector, data[:-1].T), axis=1)  # Creates matrix X, where each new line is a different training example, where x0 on each line is 1

iterations = 100000

'''Function to calculate cost for current theta values'''


def calc_cost(X, y, theta):

    m = len(y)

    predictions = X.dot(theta)  # Vector is generated where each row is a predicted value for each training example
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))  # Finds the cost associated with each row of y and predictions and adds them all together to give the total cost.
    return cost


'''Function to perform gradient descent, returns theta history, cost history, and most optimal theta values'''


def gradient_descent(X,y, theta, learning_rate, iterations):

    n = theta.shape[0] - 1  # Number of features
    m = len(y)  # Number of training examples

    cost_history = np.zeros(iterations)  # Array used to store cost associated with each iteration
    theta_history = np.zeros((iterations, n+1)) # Array used to store theta associated with each iteration

    for j in range(0, iterations):  # Each time this loop runs theta is updated (optimised a bit more).
        predictions = X.dot(theta)  # h values, m dimensional vector
        theta = theta - (1/m) * learning_rate * X.T.dot(predictions - y) # update theta
        theta_history[j, :] = theta.T  # Add new theta to theta_history
        cost_history[j] = calc_cost(X, y, theta)  # Add new cost to cost_history

    return cost_history, theta_history, theta


'''Function to scale features to allow for quicker gradient descent'''


def scale_features(X):
    X_std = X[:, 1:].std(0, keepdims=True)  # find standard deviation of each feature
    X_mean = X[:, 1:].mean(0, keepdims=True)  # find mean of each feature
    X[:, 1:] = (X[:, 1:] - X_mean) / X_std  # Normalise x1, x2, ..., xn
    return X, X_std, X_mean  # X is normalised except x0 (1)


'''Function to unscale theta values after normalising (scaling) X'''

def unscale_theta(theta, X_std, X_mean):

    theta[0,0] = theta[0, 0] - (X_mean / X_std).dot(theta[1:, 0]) # Unscale theta0
    theta[1:, 0] = theta[1:, 0] / X_std  # Unscale theta1, theta2, ..., theta n
    return theta


X_scaled, X_std, X_mean = scale_features(X)

cost_history, theta_history, theta = gradient_descent(X_scaled, y, theta, 0.0001, iterations)

iterationsX = np.arange(1, iterations + 1, dtype= int)

print("Theta is", unscale_theta(theta, X_std, X_mean), " (After unscaling)")

# Plot graph of cost vs iterations
plt.plot(iterationsX, cost_history, '-r')  # need x and y values separate or pyplot autogenerates x values.
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.xlim((1, iterations))
plt.show()