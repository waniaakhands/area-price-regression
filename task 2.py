# TASK 2
# Area vs Price
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


area = np.array([500,800,1000,1200,1500,1800,2000,2200,2500,3000]).reshape(-1,1)
price = np.array([100000,150000,200000,250000,300000,350000,400000,450000,500000,600000])

model2 = LinearRegression()
model2.fit(area, price)

pred2 = model2.predict([[2500]])

print("\nTask 2 Prediction (2500 sq.ft):", pred2[0])

mae2 = mean_absolute_error(price, model2.predict(area))
mse2 = mean_squared_error(price, model2.predict(area))

print("MAE:", mae2)
print("MSE:", mse2)

plt.scatter(area, price)
plt.plot(area, model2.predict(area))
plt.title("Area vs Price")
plt.show()