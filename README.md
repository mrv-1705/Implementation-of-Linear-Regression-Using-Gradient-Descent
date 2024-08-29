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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data, skipping non-numeric rows or columns if necessary
try:
    data = pd.read_csv("city_profit_data.csv", header=None)
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric, set non-numeric as NaN
    data = data.dropna()  # Drop rows with any NaN values
except Exception as e:
    print(f"Error loading data: {e}")

# Scatter plot of the data
plt.scatter(data[0], data[1])
plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()

# Feature Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=[0, 1])  # Convert back to DataFrame for easier handling

# Prepare the data for linear regression
data_n = data_scaled.values
m = data_n[:, 0].size
X = np.append(np.ones((m, 1)), data_n[:, 0].reshape(m, 1), axis=1)
y = data_n[:, 1].reshape(m, 1)
theta = np.zeros((2, 1))

# Ensure X, y, and theta are NumPy arrays
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)
theta = np.array(theta, dtype=float)

# Debugging prints
print("Initial X:")
print(X)
print("Initial y:")
print(y)
print("Initial theta:")
print(theta)

# Define the cost function
def computeCost(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)  # Ensure dot product is used correctly
    square_err = (h - y) ** 2
    return 1 / (2 * m) * np.sum(square_err)

# Define the gradient descent function
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []  # Empty list to store cost over iterations
    for i in range(num_iters):
        predictions = np.dot(X, theta)
        error = np.dot(X.T, (predictions - y))
        descent = alpha * (1 / m) * error
        theta -= descent

        # Debugging prints to trace values
        if i % 100 == 0:  # Print every 100 iterations
            print(f"Iteration {i}: theta = {theta.ravel()}")

        J_history.append(computeCost(X, y, theta))
    return theta, J_history

# Run gradient descent with a smaller learning rate
alpha = 0.001  # Reduced learning rate
num_iters = 1500
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Debugging prints
print("Final theta:")
print(theta)

# Print the hypothesis equation
print(f"h(x) = {round(theta[0, 0], 2)} + {round(theta[1, 0], 2)}x1")

# Plot the cost function history
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

# Plot the linear regression line on top of the scatter plot
plt.scatter(data[0], data[1])
x_value = np.linspace(min(data[0]), max(data[0]), 100)  # Create more data points for a smooth line
y_value = [float(theta[0]) + float(theta[1]) * xi for xi in x_value]  # Ensure correct float conversion
plt.plot(x_value, y_value, color="r")
plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()

# Prediction function
def predict(x, theta):
    x = np.array(x, dtype=float)  # Ensure x is a NumPy array
    return float(np.dot(x, theta))  # Ensure correct dot product and float conversion

# x1 = scaler.transform([3.5])[0, 0]  # Scale input features
predict1 = predict([1,3.5], theta) * 10000
print(f"For Population = 35000, we predict a profit of ${round(predict1, 0)}")

# x2 = scaler.transform([[7]])[0, 0]  # Scale input features
predict2 = predict([1, 7], theta) * 10000
print(f"For Population = 70000, we predict a profit of ${round(predict2, 0)}")

```

## Output:
![download](https://github.com/user-attachments/assets/1f187abe-c863-45d9-b88c-3ad6a6d78514)
![download](https://github.com/user-attachments/assets/1d98a48b-f7d9-42de-bdd2-f2e59744e2b7)
![download](https://github.com/user-attachments/assets/70ee20a6-eef1-4af5-913f-c305e98bf7d4)
![image](https://github.com/user-attachments/assets/6ce699ff-830b-41f5-838f-f87bc4fcdc78)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
