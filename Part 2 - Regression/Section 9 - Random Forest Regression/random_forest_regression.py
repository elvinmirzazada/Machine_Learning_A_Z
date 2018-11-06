import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#_-----------------------Fitting Random Forest regression
from sklearn.ensemble import RandomForestRegressor
rand_reg = RandomForestRegressor(n_estimators = 300, criterion = 'mse', random_state = 0)
rand_reg.fit(X, y)
acc = rand_reg.score(X, y)    
y_pred = rand_reg.predict((6.5))
#y_pred = sc_y.inverse_transform(y_pred)


plt.scatter(X, y, color='red')
plt.plot(X, rand_reg.predict(X), color='blue')
plt.show()

