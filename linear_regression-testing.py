import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#Here creating arrays of x, y
time_studied = np.array([5, 15, 25, 35, 45, 55]).reshape(-1,1)
scores = np.array([5, 20, 14, 32, 22, 38]).reshape(-1,1)

time_train, time_test, score_train, score_test = train_test_split(time_studied,scores,test_size=0.5)

model = LinearRegression()
#Fit x,y to model
model.fit(time_studied, scores)
print(f"Score: {model.score(time_test,score_test)*100} %")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")


plt.scatter(time_studied, scores)
plt.plot(np.linspace(0,60,100).reshape(-1,1), model.predict(np.linspace(0,60,100).reshape(-1,1)), "r")
plt.ylim(0,100)
plt.show()
