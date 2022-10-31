# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:46:31 2022

@author: alihsan-tsdln
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("winequality-red.csv", sep=";")

inputs_data = dataset.iloc[:, :11]
outputs_data = dataset.iloc[:, 11:]

outputs_data = StandardScaler().fit_transform(outputs_data)
outputs_data = pd.DataFrame(data = outputs_data)
   
inputs_data = pd.DataFrame(data = PolynomialFeatures(degree = 3).fit_transform(inputs_data, outputs_data))

inputs_data = StandardScaler().fit_transform(inputs_data)
inputs_data = pd.DataFrame(data = inputs_data)

model = sm.OLS(outputs_data.values, inputs_data.values).fit()
b = model.pvalues
c = np.argmax(b)


while b[c] > 0.05:
    model = sm.OLS(outputs_data.values, inputs_data.values).fit()
    b = model.pvalues
    c = np.argmax(b)
    inputs_data = inputs_data.drop(inputs_data.columns[c], axis = 1) 

x_test, x_train, y_test, y_train = train_test_split(inputs_data, outputs_data, train_size=0.1, random_state=42)


rfr = RandomForestRegressor(n_estimators=50, random_state=42).fit(x_train.values,y_train.values.ravel())
predict = rfr.predict(x_test.values)

success = r2_score(y_test.values, predict)
print(success)


#0.5695 | 0.05 > p | Random Forest












