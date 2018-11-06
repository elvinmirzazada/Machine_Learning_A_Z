import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import datas
dataset = pd.read_csv('data/50_Startups.csv')
dataset = pd.get_dummies(dataset, columns=['State'], drop_first=True)


X = dataset.drop(['Profit'], axis=1).values
y = dataset.loc[:, 'Profit'].values

#Splitting datas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0, shuffle = True)
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Fitting Multiple Linear Regression on training and test set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs = -1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
acc = regressor.score(X_test, y_test)

#Compare Prediction and actual values
plt.plot(y_test, color = 'r')
plt.plot(y_pred, color = 'g')
plt.show()

#--------------------Building the optimal model using Backward Elimination-------------------------------------

import statsmodels.formula.api as sm
X = np.append(values = X, arr = np.ones((50, 1)).astype(int), axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#def backwardElimination(x, sl):
#    numVars = len(X[0])
#    for i in range(0, numVars):
#        



X_train, X_test, y_train, y_test = train_test_split(X[:, [1]], y, test_size = 0.2, random_state = 0, shuffle = True)
#Fitting Multiple Linear Regression on training and test set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs = -1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
acc = regressor.score(X_test, y_test)

#Compare Prediction and actual values
plt.plot(y_test, color = 'r')
plt.plot(y_pred, color = 'g')
plt.show()














