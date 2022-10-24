# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 23:13:50 2022

@author: alihsan-tsdln
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df = pd.read_csv("maaslar.csv")


inputs = df.iloc[:, 1:2]
outputs = df.iloc[:, 2:]

inputs = StandardScaler().fit_transform(inputs)
outputs = StandardScaler().fit_transform(outputs)

plt.scatter(inputs, outputs, color = "blue")

#svr is Support Vector Regression 
#rbf is Radial Basis Function
svr_rbf = SVR(kernel="rbf").fit(inputs, outputs)
plt.plot(inputs, svr_rbf.predict(inputs))





