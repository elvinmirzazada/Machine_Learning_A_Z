import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 0, shuffle = True)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""

#_-----------------------Fitting Linear Regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)
acc = linear_reg.score(X, y)               #66%

#-----------------------Fitting Polynomial Regression 
from sklearn.preprocessing import PolynomialFeatures
polynmial = PolynomialFeatures(degree=8)
X_poly = polynmial.fit_transform(X)


from sklearn.linear_model import LinearRegression
poly_linear_reg = LinearRegression()
poly_linear_reg.fit(X_poly, y)
acc = poly_linear_reg.score(X_poly, y)    #100%

y_pred = poly_linear_reg.predict(X_poly)


#----------------------Visualising Linear Regression results
plt.scatter(X, y, c='r')
plt.plot(X, linear_reg.predict(X))
#plt.show()


#----------------------Visualising Polynomial Regression results
#plt.scatter(X, y)
plt.plot(X, poly_linear_reg.predict(polynmial.fit_transform(X)), c='blue')
plt.show()


























