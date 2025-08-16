# Machine Learning

## Simple and Logistic Regression
- Models linear relationship between continuous target variable and explanatory features
- Can be used to predict a continuous value
- A single independent variable estimates dependent variable
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

df = data.read_csv('link')

# simple scatter graph
plt.scatter(df['x-axis'],df['y-axis'],data=df,color='blue')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

# simple regression line
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df['x-axis'],df['y-axis'],test_size=int,random_state=int)

from sklearn import linear_model

regressor = linear_model.LinearRegression()

regressor.fit(x_train.reshape(-1,1),y_train)

# scatch the regression line
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,regressor.coef_*X_train+regressor.intercept_,'-r')
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# model evaluation
from sklearn metrics import mean_absolute_error, mean_squared_error, r2_score

yhat = regressor.predict(x_test.reshape(-1,1))

mean_absolute_error(y_test,yhat)
mean_squared_error(y_test,yhat)
r2_score(y_test,yhat)
```

##
