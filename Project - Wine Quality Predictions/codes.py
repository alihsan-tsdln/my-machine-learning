# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:29:07 2022

@author: alihsan-tsdln
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("winequality-red.csv", sep=";")

inputs_data = dataset.iloc[:, :11]
outputs_data = dataset.iloc[:, 11:]

inputs_data = np.append(arr = np.ones((len(inputs_data.index),1)).astype(int), values = inputs_data, axis = 1)
inputs_data = pd.DataFrame(data = inputs_data)

#p < 0.05
inputs_data = inputs_data.drop(inputs_data.columns[8], axis = 1)
inputs_data = inputs_data.drop(inputs_data.columns[4], axis = 1)
inputs_data = inputs_data.drop(inputs_data.columns[1], axis = 1)
inputs_data = inputs_data.drop(inputs_data.columns[2], axis = 1)
model = sm.OLS(outputs_data, inputs_data).fit()
print(model.summary())
np.argmax(model.pvalue)

# I tried to understanding to effect of random states

maxVal = 0
maxSize = 0
maxRand = 0

for i in np.arange(0.33, 1.0, 0.33):
    for j in range(50):
        x_test, x_train, y_test, y_train = train_test_split(inputs_data, outputs_data, train_size=i, random_state=j)
        
        for k in range(50):
            predict = RandomForestRegressor(random_state=k).fit(x_train.values, y_train.values.ravel()).predict(x_test.values)
            a = (len(x_train.index) - 1) / (len(x_train.index) - len(x_train.columns) - 1)
            b = 1 - r2_score(y_test.values, predict)
            scsT = 1 - (a * b)
            
            print(j, end = " ")
            
            if scsT > maxVal:
                maxVal = scsT
                maxSize = i
                maxRabd = j
                print("Max Value Size                :: " + str(i))
                print("Max Value Random              :: " + str(j))
                print("Max Forest Random             :: " + str(k))
                print("Random Forest Regressor       :: " + str(scsT) + "\n")
                predict = RandomForestRegressor().fit(x_train.values, y_train.values.ravel()).predict(x_train.values)
                a = (len(x_train.index) - 1) / (len(x_train.index) - len(x_train.columns) - 1)
                b = 1 - r2_score(y_train.values, predict)
                scsT = 1 - (a * b)
                print("Max Value Size Train          :: " + str(i))
                print("Max Value Random Train        :: " + str(j))
                print("Max Forest Random Train       :: " + str(k))
                print("Random Forest Regressor Train :: " + str(scsT) + "\n\n\n")




# Why values are so perfect like that

# Max Value Size                :: 0.01
# Max Value Random              :: 4
# Random Forest Regressor       :: 0.874639306350721

# Max Value Size Train          :: 0.01
# Max Value Random Train        :: 4
# Random Forest Regressor Train :: 0.9304304131398792
    

# Max Value Size                :: 0.005
# Max Value Random              :: 2
# Random Forest Regressor       :: 0.8748262289029537



# Max Value Size Train          :: 0.005
# Max Value Random Train        :: 2
# Random Forest Regressor Train :: 0.9305334341937499


#test have 7 row, train have 1592

# Max Value Size                :: 0.0015
# Max Value Random              :: 0
# Random Forest Regressor       :: 0.9609307255520505



# Max Value Size Train          :: 0.0015
# Max Value Random Train        :: 0
# Random Forest Regressor Train :: 0.9294093703797117
    

#test have 15 row, train have 1585

# Max Value Size                :: 0.003
# Max Value Random              :: 42
# Random Forest Regressor       :: 0.9931024005053696



# Max Value Size Train          :: 0.003
# Max Value Random Train        :: 42
# Random Forest Regressor Train :: 0.9301214215570631



#0.015 is done. 2 test value
 
# Max Value Size                :: 0.0015
# Max Value Random              :: 13
# Max Forest Random             :: 32
# Random Forest Regressor       :: 0.9973819558359621

# Max Value Size Train          :: 0.0015
# Max Value Random Train        :: 13
# Max Forest Random Train       :: 32
# Random Forest Regressor Train :: 0.9314257096846646
    
    
    
    
    
    
    