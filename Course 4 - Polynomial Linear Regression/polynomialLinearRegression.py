import pandas as pd
data = pd.read_csv("maaslar.csv")

input_data = data.iloc[:, 1:2]
output_data = data.iloc[:, 2:]

import matplotlib.pyplot as plt
plt.scatter(input_data, output_data, color = "blue")

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
import numpy as np

colors = np.array(["grey", "green", "purple", "red", "orange", "brown", "black"])

for i in range(1,7):
    pf = PolynomialFeatures(degree = i).fit_transform(input_data)
    a = predictDeg = LinearRegression().fit(pf, output_data)
    predictDeg = LinearRegression().fit(pf, output_data).predict(pf)
    plt.plot(input_data,predictDeg, color = colors[i])

