# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 23:11:14 2022

@author: alihsan-tsdln
"""

import pandas as pd

df = pd.read_csv("satislar.csv")

months = df.iloc[:, :1]
sales = df.iloc[:, 1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
guess = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

import matplotlib.pyplot as plt
plt.plot(x_train, y_train)
plt.plot(x_test, guess)
 