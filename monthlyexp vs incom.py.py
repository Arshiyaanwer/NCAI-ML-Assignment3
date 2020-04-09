# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:11:34 2020

@author: Arshiya Anwer

"""
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/monthly Experience Vs Income/monthlyexp vs incom.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, -1].values
print('Monthly Experience')
print(X)
print('Income')
print(y)

# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression
linear_Regressor= LinearRegression()
linear_Regressor.fit(X,y)

# Predicting the Test set results
y_pred= linear_Regressor.predict(X_test)

# Mapping the Training set results
plt.scatter(X_train, y_train, color = 'black')
plt.plot(X_train,linear_Regressor.predict(X_train), color = 'red')
print('Linear Regression')
plt.title('Monthly Experience Vs Income (Training set)')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.show()

# Mapping the Test set results
plt.scatter(X_test, y_test, color = 'black')
plt.plot(X_train, linear_Regressor.predict(X_train), color = 'red')
plt.title('Monthly Experience Vs Income (Test set)')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.show()

print('We did not get the Best-fit line')
print('Now Applying Polynomial Regression')
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor= PolynomialFeatures(degree= 6)
X_poly = polynomial_regressor.fit_transform(X)
polynomial_regressor.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# mapping the Polynomial Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg_2.predict(polynomial_regressor.fit_transform(X)), color = 'red')
plt.title('Monthly Experience Vs Income (Polynomial Regression)')
plt.xlabel('Monthly income')
plt.ylabel('Income')
plt.show()

