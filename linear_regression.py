import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
#Here creating arrays of x, y
time_studied = np.array([5, 15, 25, 35, 45, 55]).reshape(-1,1)
scores = np.array([5, 20, 28, 40, 22, 38]).reshape(-1,1)
#Precreated model from Sklearn module
model = LinearRegression()
#Fit x,y to model
model.fit(time_studied, scores)
#Matplotlib scatter for x,y then drawing the red line of the predicted values in red color
plt.scatter(time_studied, scores)
plt.plot(np.linspace(0,60,100).reshape(-1,1), model.predict(np.linspace(0,60,100).reshape(-1,1)), "r")
plt.ylim(0,100)
plt.show()
