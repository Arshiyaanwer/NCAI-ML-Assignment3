"""
Created on Fri Apr  3 20:17:54 2020

@author: Arshiya Anwer
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#for NewYork
# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/ml/50 startups/50_Startups.csv')

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values 


#converting text "state" into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

X1 = dataset.loc[dataset.State=="New York","State":"Profit"]
print(X1)

yNY= X1.iloc[:,-1].values
XNY= np.arange(17).reshape(-1,1)
print("Profit")
print(yNY)
print("Startup Expendeture")
print(XNY)

#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
XNY_train, XNY_test, yNY_train, yNY_test = train_test_split(XNY, yNY, test_size = 0.2, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(XNY, yNY)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =5)
XNY_poly = poly_reg.fit_transform(XNY)
poly_reg.fit(XNY_poly, yNY)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(XNY_poly, yNY)

# Visualising the Linear Regression results
plt.scatter(XNY, yNY, color = 'blue')
plt.plot(XNY, lin_reg.predict(XNY), color = 'black')
plt.title('(NewYork Startups)')
plt.xlabel('startup')
plt.ylabel('profit')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(XNY, yNY, color = 'blue')
plt.plot(XNY, lin_reg_1.predict(poly_reg.fit_transform(XNY)), color = 'black')
plt.title('NewYork Startups')
plt.xlabel('startups')
plt.ylabel('profit')
plt.show()





 
#for California
# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/ml/50 startups/50_Startups.csv')

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values 


#converting text "state" into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_Xx = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

X2 = dataset.loc[dataset.State=="California",'State':'Profit']
print(X2)

yC= X2.iloc[:,-1].values
XC= np.arange(17).reshape(-1,1)
print("Profit")
print(yC)
print('startups')
print(XC)

#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
XC_train, XC_test, yC_train, yC_test = train_test_split(XC, yC, test_size = 0.2, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regC = LinearRegression()
lin_regC.fit(XC, yC)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =5)
XC_poly = poly_reg.fit_transform(XC)
poly_reg.fit(XC_poly, yC)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(XC_poly, yC)

# Visualising the Linear Regression results
plt.scatter(XC, yC, color = 'blue')
plt.plot(XC, lin_regC.predict(XC), color = 'black')
plt.title('(California Startups')
plt.xlabel('startups')
plt.ylabel('profit')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(XC, yC, color = 'blue')
plt.plot(XC, lin_reg_2.predict(poly_reg.fit_transform(XC)), color = 'black')
plt.title('California Startups')
plt.xlabel('startups')
plt.ylabel('profit')
plt.show()

# Predicting future Profit (NEWYORK)

a=lin_reg_1.predict(poly_reg.fit_transform([[25]]))
print("Prediction Of Future Profit (NEWYORK")
print(a)
 
#Predicting future Profit (CALIFORNIA)
print("Prediction Of Future Profit (CALIFORNIA)")
b=lin_reg_2.predict(poly_reg.fit_transform([[25]]))
print(b)

