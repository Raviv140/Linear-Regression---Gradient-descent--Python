import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

x = sy.Symbol('x')
df = pd.read_csv('USA_Housing.csv')
print(df.info())
# sns.jointplot(x='Avg. Area Number of Rooms', y='Price', data=df.head(100))
X = np.asarray(df['Avg. Area Number of Rooms'])
y = np.asarray(df['Price']) / 1000
theta0_coeff = 0
theta1_slope = 10
yy = X * theta1_slope + theta0_coeff

plt.xlabel('Number of Rooms')
plt.ylabel('Price in US $k ')
plt.plot(X, y, '.b', X, yy, 'r')
plt.pause(3)
alpha = 0.01
m = X.size

for i in range(10):

    def sigma_0(xarray, yarray, th0, th1):
        S = 0
        for j in range(xarray.size):
            S += (th1 * xarray[j] + th0) - yarray[j]
        return S


    def sigma_1(xarray, yarray, th0, th1):
        S = 0
        for j in range(xarray.size):
            S += ((th1 * xarray[j] + th0) - yarray[j]) * xarray[j]
        return S


    temp0 = theta0_coeff - alpha * (1 / m) * sigma_0(X, y, theta0_coeff, theta1_slope)
    temp1 = theta1_slope - alpha * (1 / m) * sigma_1(X, y, theta0_coeff, theta1_slope)
    theta0_coeff = temp0
    theta1_slope = temp1
    plt.plot(X, y, '.b', X, yy, 'w')
    yy = theta1_slope * X + theta0_coeff
    plt.plot(X, y, '.b', X, yy, 'r')
    plt.pause(1)

f = round(theta1_slope, 3) * x + round(theta0_coeff, 3)
print(f" The linear Regression Line is : f(x) = {f} ")
y_new = theta1_slope * X + theta0_coeff
plt.plot(X, y, '.b', X, y_new, 'r')
plt.text(3, 2100, f" The linear Regression Line is :\n f(x) = {f} ", color='black', alpha=5)
plt.show()
