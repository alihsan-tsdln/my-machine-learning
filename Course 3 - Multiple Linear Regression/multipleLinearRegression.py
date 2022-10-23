# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:13:59 2022

@author: alihsan-tsdln
"""

import pandas as pd

df = pd.read_csv("data.csv")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ulke = df.iloc[:, :1]
cinsiyet = df.iloc[:, 4:]

ulke.iloc[:,0] = le.fit_transform(ulke)
cinsiyet.iloc[:, 0] = le.fit_transform(cinsiyet)
cinsiyet.columns = ["Kadin"]

oh = OneHotEncoder()
ulke = pd.DataFrame(data = oh.fit_transform(ulke).toarray(), columns = ["fr","tr","us"])

input_datas = pd.concat([ulke, df.iloc[: , 1:4]], axis=1)

from sklearn.impute import SimpleImputer
import numpy as np

si = SimpleImputer(missing_values=np.nan, strategy="mean")
si = si.fit(input_datas.iloc[:, 5:])
input_datas.iloc[:, 5:] = si.transform(input_datas.iloc[:, 5:])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(input_datas,cinsiyet, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr = lr.fit(x_train, y_train)

sex_guess = lr.predict(x_test)

input2_datas= pd.concat([input_datas.drop(["boy"], axis = 1), cinsiyet], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(input2_datas,input_datas.iloc[:, 3:4], test_size=0.2, random_state=42)

lr = LinearRegression()
lr = lr.fit(x_train, y_train)

height_guess = lr.predict(x_test)

import statsmodels.api as sm

models = np.array(input2_datas, dtype=np.float32)
boy = np.array(input_datas.iloc[:, 3:4], dtype=np.float32)

model = sm.OLS(boy, models).fit()
print(model.summary())

##p-value of x5(4.column) is bigger than 0.05 so we must remove it

models = np.delete(models, 4, 1)
model = sm.OLS(boy, models).fit()
print(model.summary())

##p-value of x5(4.column) is smaller than 0.05 so we must not remove it and I didnt prefer remove








