# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:59:47 2020

@author: Tasmiya Anwer
"""
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/Housing Price/housing price.csv')
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree =3)
X_poly = polynomial_regressor.fit_transform(X)
polynomial_regressor.fit(X_poly, Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X,linear_regressor.predict(X), color = 'blue')
plt.title('housing Price According to ID (linear Regression')
plt.xlabel('ID')
plt.ylabel('Selling Price')

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict( polynomial_regressor.fit_transform(X)), color = 'blue')
plt.title('housing price according to ID(polynomial Regression)')
plt.xlabel('ID')
plt.ylabel('Selling Price')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(polynomial_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('housing price according to ID(polynomial Regression')
plt.xlabel('ID')
plt.ylabel('Selling Price')
plt.show()

# Predicting a new result with Linear Regression
print('The Selling Price Of the House with ID 2670')
a=linear_regressor.predict([[2670]])
print(a)

# Predicting a new result with Polynomial Regression
print('The selling Price of the house with ID 3500')
b=lin_reg_2.predict(polynomial_regressor.fit_transform([[3500]]))
print(b)
