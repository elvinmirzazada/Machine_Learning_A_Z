import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 0, shuffle = True)

#Simple Linear Regression training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train, y_train)

#Predict test set
y_pred = regressor.predict(X_test)
score = regressor.score(X_test, y_test)

#Visualize Training and testing set result
plt.scatter(X_train, y_train, color = 'r')
plt.plot(X_train, regressor.predict(X_train), color ='b')

plt.scatter(X_test, y_test, color = 'g')
plt.show()
















