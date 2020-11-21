import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("monthly_csv.csv")
data.head()

x = np.arange(len(data))
y = data["Price"].to_numpy()
x = x.reshape(-1,1)
y = y.reshape(-1,1)


reg = LinearRegression()
reg.fit(x, y)
reg.score(x,y)
reg.predict([[1050]]).item()

plt.plot(x, reg.predict(x))
plt.scatter(x, y, c="r")