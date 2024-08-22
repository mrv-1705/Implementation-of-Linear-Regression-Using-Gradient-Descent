# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load Data and Prepare**: Load the dataset, extract features (X) and target (y), and add a bias term to X.
2. **Initialize Parameters**: Initialize the parameters (theta) to zero.
3. **Compute and Optimize**: Define the cost function and perform gradient descent to iteratively update theta.
4. **Predict and Visualize**: Use the optimized theta to make predictions, plot the regression line, and predict for new data.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Kurapati Vishnu Vardhan Reddy
RegisterNumber:  212223040103
```
```
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset (replace with your data)
data = np.loadtxt('data.txt', delimiter=',')  # Assuming the dataset is in a file called 'data.txt'
X = data[:, 0]  # Population
y = data[:, 1]  # Profit
m = len(y)  # Number of training examples

# Add a column of ones to X to account for the intercept term (theta_0)
X = np.column_stack((np.ones(m), X))

# 2. Initialize parameters (theta_0 and theta_1)
theta = np.zeros(2)  # [theta_0, theta_1]

# 3. Cost function
def compute_cost(X, y, theta):
    predictions = X @ theta  # Predictions of hypothesis on all m examples
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(np.square(error))  # Mean squared error
    return cost

# 4. Gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = []  # To store the cost at each iteration

    for i in range(num_iters):
        predictions = X @ theta
        error = predictions - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - alpha * gradient  # Update the parameters

        J_history.append(compute_cost(X, y, theta))  # Save the cost at each iteration

    return theta, J_history

# Parameters for gradient descent
alpha = 0.01  # Learning rate
iterations = 1500

# Run gradient descent
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# 5. Make predictions and plot the results
plt.scatter(X[:, 1], y, label='Training Data')  # Plot the data
plt.plot(X[:, 1], X @ theta, label='Linear Regression', color='red')  # Plot the linear fit
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend()
plt.show()

# Predict the profit for a city with a population of 35,000
population = 3.5  # Since the feature is scaled (Population in 10,000s)
predicted_profit = np.array([1, population]) @ theta
print(f'Predicted profit for a city with population of 35,000: ${predicted_profit * 10000:.2f}')
```

## Output:
![image](https://github.com/user-attachments/assets/8748646e-0088-402f-855f-cf94d2e54821)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
