import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

#_-----------------------Fitting Decision Tree
from sklearn.tree import DecisionTreeRegressor
decision_reg = DecisionTreeRegressor(random_state = 0)
decision_reg.fit(X, y)
acc = decision_reg.score(X, y)    
y_pred = decision_reg.predict(sc_x.transform(6.6))
y_pred = sc_y.inverse_transform(y_pred)


plt.scatter(X, y, color='red')
plt.plot(X, decision_reg.predict(X), color='blue')
plt.show()





























