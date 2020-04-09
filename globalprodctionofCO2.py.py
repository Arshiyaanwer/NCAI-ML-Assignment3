# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 03:42:02 2020

@author: Arshiya Anwer

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/global CO2/global_co2.csv')
X = dataset.iloc[201:, 0:1].values
print(X)
y = dataset.iloc[201:, -1].values
print(y)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a CO2 emission in year 2011,2012,2013

a= regressor.predict([[2011]])
print('global production of CO2 in year 2011')
print(a)
b=regressor.predict([[2012]])
print(' global production of CO2 in year 2012')
print(b)
c=regressor.predict([[2013]])
print('global production of CO2 in year 2013')
print(c)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('global CO2 production (Decision Tree Regression)')
plt.xlabel('year')
plt.ylabel('CO2 produce globally per capita')
plt.show()

