# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:52:31 2022

@author: alihsan-tsdln
"""

import numpy as np
import pandas as pd

df = pd.read_csv("data1.csv")

yas = df.iloc[:, 1:4].values

from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy="mean")
impute = impute.fit(yas)
yas = impute.transform(yas)

ulke = df.iloc[:, 0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(ulke)

ho = preprocessing.OneHotEncoder()
ulke = ho.fit_transform(ulke).toarray()

input_data = pd.concat([pd.DataFrame(data = ulke, columns = ["fr","tr","us"])
                       ,pd.DataFrame(data = yas, columns=["boy","kilo","yas"])]
                       , axis = 1)

output_data = pd.DataFrame(data=df.iloc[:,4])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

scaled_x_test = sc.fit_transform(x_test)
scaled_x_train = sc.fit_transform(x_train)

























