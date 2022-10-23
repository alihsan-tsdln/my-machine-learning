# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:06:35 2022

@author: alihsan-tsdln
"""

import pandas as pd

df = pd.read_csv("datas.csv")
outlooks = df.iloc[:, 0:1] 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
outlooks = pd.DataFrame(data = le.fit_transform(outlooks))

oh = OneHotEncoder()
outlooks = pd.DataFrame(data = oh.fit_transform(outlooks).toarray(), columns = ["overcast", "rainy", "sunny"])

plays = df.iloc[:, 4:]
le = LabelEncoder()
plays = pd.DataFrame(data = le.fit_transform(plays), columns = ["Play"])

windys = df.iloc[:, 3:4]
le = LabelEncoder()
windys = pd.DataFrame(data = le.fit_transform(windys), columns = ["Windy"])


input_datas = pd.concat([outlooks, windys, df.iloc[:, 1:3]], axis = 1)

import statsmodels.api as sm
import numpy as np

input_datas = input_datas.iloc[:, [0,1]]

models = np.array(input_datas, dtype=np.float32)
play_np = np.array(plays, dtype=np.float32)

model = sm.OLS(play_np, models).fit()
print(model.summary())

from sklearn.model_selection import train_test_split

x_test, x_train, y_test, y_train = train_test_split(input_datas, plays, train_size = 0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr = lr.fit(x_train, y_train)
guess = lr.predict(x_test)















