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
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #Calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("50_Startups.csv")
data.head()

#Assuming rhe last column is your target variable 'y' and the preceding columns.
X = (data.iloc[1:,:-2].values)
X1 =X.astype(float)

scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target calue for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")


```

## Output:
![download](https://github.com/user-attachments/assets/1f187abe-c863-45d9-b88c-3ad6a6d78514)
## Cost Function:
![image](https://github.com/user-attachments/assets/e5f862f1-a2ed-40c7-9783-ae9bbcdfe156)

![download](https://github.com/user-attachments/assets/1d98a48b-f7d9-42de-bdd2-f2e59744e2b7)
![download](https://github.com/user-attachments/assets/70ee20a6-eef1-4af5-913f-c305e98bf7d4)
![image](https://github.com/user-attachments/assets/6ce699ff-830b-41f5-838f-f87bc4fcdc78)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
