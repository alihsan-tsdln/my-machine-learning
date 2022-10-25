# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:16:22 2022

@author: alihsan-tsdln
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("maaslar.csv")

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:]

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0).fit(x.values, y.values)
plt.scatter(x.values, y.values, color = "blue")
plt.plot(x.values, dtr.predict(x.values), color = "red")