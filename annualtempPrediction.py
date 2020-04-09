# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:12:56 2020

@author: Tasmiya Anwer
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/ml/annual tempreture/annual_temp.csv')
A = dataset.iloc[:, 1:2].values
B = dataset.iloc[:, -1].values
 
X1 = dataset.loc[dataset.Source=="GCAG",:]
print(X1)

y= X1.iloc[:,-1].values
X= X1.iloc[:,1].values.reshape(-1,1)
print(y)
print(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Annual Temperature of GCAG (Linear Regression)')
plt.xlabel('year')
plt.ylabel('mean tempreture')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Annual Temperature of GCAG (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('mean Temperature')
plt.show()

print('line doesnt fit')

#desicion tree Regression
# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/ml/annual tempreture/annual_temp.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

X1 = dataset.loc[dataset.Source=="GCAG",:]
print(X1)

y= X1.iloc[:,-1].values
X= X1.iloc[:,1].values.reshape(-1,1)
print(y)
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)



# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title(' Annual Temperature of GCAG(Decision Tree Regression)')
plt.xlabel('year')
plt.ylabel('mean temperature')
plt.show()


#FOR GISTEMP

# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/ml/annual tempreture/annual_temp.csv')
C = dataset.iloc[:, 1:2].values
D = dataset.iloc[:, -1].values
 
X2 = dataset.loc[dataset.Source=="GISTEMP",:]
print(X1)

ygis= X1.iloc[:,-1].values
Xgis= X1.iloc[:,1].values.reshape(-1,1)
print(ygis)
print(Xgis)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xgis_train, Xgis_test, ygis_train, ygis_test = train_test_split(Xgis, ygis, test_size = 1/3, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(Xgis, ygis)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg_gis = PolynomialFeatures(degree =5)
Xgis_poly = poly_reg_gis.fit_transform(Xgis)
poly_reg.fit(Xgis_poly, ygis)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(Xgis, ygis, color = 'red')
plt.plot(Xgis, lin_reg_1.predict(X), color = 'blue')
plt.title('Annual Temperature of gistemp (Linear Regression)')
plt.xlabel('year')
plt.ylabel('mean tempreture')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(Xgis, lin_reg_3.predict(poly_reg_gis.fit_transform(Xgis)), color = 'blue')
plt.title('Annual Temperature of Gistemp (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('mean Temperature')
plt.show()

print('line doesnt fit')

#desicion tree Regression

# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/ml/annual tempreture/annual_temp.csv')
C= dataset.iloc[:, 1:2].values
D = dataset.iloc[:, -1].values

X2 = dataset.loc[dataset.Source=="GISTEMP",:]
print(X2)

ygis= X2.iloc[:,-1].values
Xgis= X2.iloc[:,1].values.reshape(-1,1)
print(ygis)
print(Xgis)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xgis_train, Xgis_test, ygis_train, ygis_test = train_test_split(Xgis, ygis, test_size = 0.01, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_gis = DecisionTreeRegressor(random_state = 0)
regressor_gis.fit(Xgis, ygis)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid_1 = np.arange(min(Xgis), max(Xgis), 0.01)
X_grid_1 = X_grid_1.reshape((len(X_grid_1), 1))
plt.scatter(Xgis, ygis, color = 'red')
plt.plot(X_grid_1, regressor.predict(X_grid_1), color = 'blue')
plt.title(' Annual Temperature of Gistemp(Decision Tree Regression)')
plt.xlabel('year')
plt.ylabel('mean temperature')
plt.show()

# Predicting a new result
a = regressor.predict([[2016]])
print('Mean Tempreture of GCAG in 2016 will be:')
print(a)
b=regressor.predict([[2017]])
print('Mean Tempreture of GCAG in 2017 will be:')
print(b)
c=regressor_gis.predict([[2016]])
print('Mean Tempreture of GISTEMP in 2016 will be:')
print(c)
d=regressor_gis.predict([[2017]])
print('Mean Tempreture of GCAG in 2017 will be:')
print(d)