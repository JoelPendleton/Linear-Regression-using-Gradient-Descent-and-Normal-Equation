import numpy as np
import matplotlib.pyplot as plt

n = 1
m = 97
theta = np.zeros((2, 1))

alpha = 0.01
iterations = 1500

x1, y = np.loadtxt("data/OneFeature.txt", delimiter=",", unpack = True)  # Ouput is 1D. Vectors are 2D m x 1 dimension so need to change to 2D using reshape

x1 = np.reshape(x1, (x1.shape[0], 1))  # Makes x1 a vector with each new line different training set
y = np.reshape(y, (y.shape[0], 1))  # Makes y a vector with each new line different training set

x1_mean = np.mean(x1)
x1_std = np.std(x1)
x1_scaled = (x1 - x1_mean) / x1_std
onesVector = np.ones((x1.shape[0], 1))

X = np.concatenate((onesVector,x1_scaled),axis=1)  # Creates matrix X, where each new line is a different training example, where x0 on each line is 1

iterationsX = np.arange(1, iterations + 1, dtype= int)
costVector = np.zeros([iterations, 1])  # Vector of cost for each iteration

for j in range(0, iterations):  # Repeat many times to optimise theta
    sumCost = 0
    sumTheta = 0
    for i in range(0, m):  # Iterate through dataset to find sigma (the cost's derivative for current theta values)
        # Change shape of feature so it's 2D and a vector
        currentFeatures = np.reshape(X[i, :], (X[i, :].shape[0], 1))  # Current features is a vector of x0 followed by x1, ... xn
        h = np.matmul(theta.transpose(), currentFeatures)  # Calculate new hypothesis based on current training set (vectorised h(x) version)

        sumCost += (h - y[i, 0]) ** 2
        sumTheta += (h - y[i, 0]) * currentFeatures  # Add on the cost functions derivative associated with this training set

    cost = (1 / (2 * m)) * sumCost  # Cost for current theta parameters
    costVector[j, 0] = cost  # Add cost for current theta parameters to costVector
    sigma = (1/m) * sumTheta
    theta += - alpha * sigma  # update theta vector to more optimal values

#unscale theta
theta[0, 0] = theta[0,0] - (x1_mean / x1_std) * theta[1, 0]
theta[1, 0] = theta[1, 0] / x1_std

print("Cost function = ", cost)
print("Theta =", theta)

# Plot graph of x against y
plt.plot(x1, y, 'rx')  # need x and y values separate or pyplot autogenerates x values.

# Plot line of best fit
xline = np.linspace(0,20,2000)
yline = theta[0, 0] + xline * theta[1, 0]
plt.plot(xline, yline, '-b')

plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# Plot graph of cost function against iterations
plt.plot(iterationsX, costVector, '-r')  # need x and y values separate or pyplot autogenerates x values.
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.xlim((1, iterations))
plt.show()
