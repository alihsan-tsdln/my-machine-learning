# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:56:27 2022

@author: alihsan-tsdln
"""

import pandas as pd

data = pd.read_csv("maaslar.csv")

x = data.iloc[:, 1:2]
y = data.iloc[:, 2:3]

#with linear regression
import matplotlib.pyplot as plt
plt.scatter(x.values,y.values, color="red")

from sklearn.linear_model import LinearRegression

guess = LinearRegression().fit(x.values, y.values)
guess = guess.predict(x.values)

plt.plot(x.values, guess)
plt.scatter(x.values, y.values, color="green")

#with polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures

pf2 = PolynomialFeatures(degree = 2)
x_poly = pf2.fit_transform(x.values)

guess_poly = LinearRegression().fit(x_poly, y.values)
guess_poly = guess_poly.predict(x_poly)

plt.plot(x.values, guess_poly, color="red")

pf3 = PolynomialFeatures(degree = 3)
x_poly = pf3.fit_transform(x.values)

guess_poly = LinearRegression().fit(x_poly, y.values)
guess_poly = guess_poly.predict(x_poly)

plt.plot(x.values, guess_poly, color="blue")

pf4 = PolynomialFeatures(degree = 4)
x_poly = pf4.fit_transform(x.values)

guess_poly = LinearRegression().fit(x_poly, y.values)
guess_poly = guess_poly.predict(x_poly)

plt.plot(x.values, guess_poly, color="purple")

pf5 = PolynomialFeatures(degree = 5)
x_poly = pf5.fit_transform(x.values)

guess_poly = LinearRegression().fit(x_poly, y.values)
guess_poly = guess_poly.predict(x_poly)

plt.plot(x.values, guess_poly, color="orange")

#most powerful predicts with 4 and numbers which bigger than 4 




