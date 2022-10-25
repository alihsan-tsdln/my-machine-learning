# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:19:19 2022

@author: alihsan-tsdln
"""

import pandas as pd

df = pd.read_csv("maaslar.csv")

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:]
x2 = x - 0.5
x3 = x + 0.5

import matplotlib.pyplot as plt

plt.scatter(x.values, y.values)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=15, random_state=0).fit(x.values,y.values)


#random forest algorithms are not like decision trees.
#decision trees cannot interpret

plt.plot(x.values, rfr.predict(x.values), color = "red")
plt.plot(x.values, rfr.predict(x2.values), color="green")
plt.plot(x.values, rfr.predict(x3.values), color="blue")










