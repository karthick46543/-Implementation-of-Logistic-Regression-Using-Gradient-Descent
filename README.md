# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load the placement dataset and preprocess the data by removing unwanted columns and converting categorical values into numeric form.
2.Separate the dataset into feature variables (X) and target variable (y) for training.
3.Train the Logistic Regression model using the sigmoid function and gradient descent algorithm.
4.Predict placement results for new students and evaluate the model accuracy. 
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RISHI BALA KARTHICK K
RegisterNumber:  212225040337
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Load the Dataset
data = pd.read_csv(r'C:\Users\acer\Downloads\Placement_Data (1).csv')

print("Original Data:")
print(data.head())

 Drop Unnecessary Columns
data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)

print("\nAfter dropping 'sl_no' and 'salary':")
print(data.head())

 Convert Categorical Columns to 'category' Type
data["gender"] = data["gender"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')
data["hsc_s"] = data["hsc_s"].astype('category')

print("\nData types after converting to 'category':")
print(data.dtypes)

  Convert Categories to Numeric Codes
data["gender"] = data["gender"].cat.codes
data["ssc_b"] = data["ssc_b"].cat.codes
data["hsc_b"] = data["hsc_b"].cat.codes
data["degree_t"] = data["degree_t"].cat.codes
data["workex"] = data["workex"].cat.codes
data["specialisation"] = data["specialisation"].cat.codes
data["status"] = data["status"].cat.codes
data["hsc_s"] = data["hsc_s"].cat.codes

print("\nData after converting categories to numeric codes:")
print(data.head())

  Separate Features (X) and Target (y)
x = data.iloc[:, :-1].values   # all columns except last
y = data.iloc[:, -1].values    # last column (status)

print("\nFeature matrix X shape:", x.shape)
print("Target vector y shape:", y.shape)

  Initialize Parameters
theta = np.random.randn(x.shape[1])  # Random initial weights for each feature

print("\nInitial theta (weights):")
print(theta)

 Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

 Define Loss Function (optional to print/check)
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
    # Added small 1e-10 to avoid log(0)

  Implement Gradient Descent
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient

        # Optional: print loss every 100 iterations
        if (i + 1) % 100 == 0:
            current_loss = loss(theta, X, y)
            print(f"Iteration {i+1}, Loss: {current_loss:.4f}")

    return theta

  Train the Model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

print("\nFinal theta (weights) after training:")
print(theta)

 Define Prediction Function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

 Make Predictions and Compute Accuracy
y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)

print("\nTraining Accuracy:", accuracy)
print("Predicted labels (first 20):")
print(y_pred[:20])

 Predict for New Students
The order of features must match 'x' columns exactly.
Example new students (values must match the encoded format used above)
xnew1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
xnew2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])

y_prednew1 = predict(theta, xnew1)
y_prednew2 = predict(theta, xnew2)

print("\nPrediction for new student 1 (0 = Not Placed, 1 = Placed):", y_prednew1[0])
print("Prediction for new student 2 (0 = Not Placed, 1 = Placed):", y_prednew2[0])

```


## Output:
![logistic regression using gradient descent](sam.png)

<img width="746" height="916" alt="Screenshot 2026-05-14 211805" src="https://github.com/user-attachments/assets/abc7bb77-2ae6-4fd8-8ff4-361207d852ca" />
<img width="573" height="330" alt="Screenshot 2026-05-14 211822" src="https://github.com/user-attachments/assets/a96edf6c-fb72-4610-afe7-a54d137ffc7d" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

