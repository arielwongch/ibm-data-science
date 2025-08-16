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

## Multiple linear regression
- Uses two or more independent variables to estimate a dependent variable
- MLR performs better than SLR, but too many variables can cause overfitting
- To improve prediction, convert categorical independent variables into numerical variables
```
import numpy as np
import matplotlib.pyplot as plt
import panadas as pd
%matplotlib inline

df = data.read_csv('link')

x = df.iloc[:,[0,1]].to_numpy()
y = df.iloc[:,[2]].to_numpy()

# preprocess data
from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
x_std = std_scaler.fit_transform(x)

# multiple regression line
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_std,y,test_size=int,random_state=int)

from sklearn import linear_model

regressor = linear_model.LinearRegression()

regressor.fit(x_train,y_train)

# plot in one dim
plt.scatter(x_train[:,0],y_train, color='blue')
plt.plot(x_train[:,0],coef_[0,0]*x_train[:,0]+intercept_[0],'-r')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()
```

## Polynomial and Non-linear Regression
- Models non-linear relationship between dependent variable and independent variables
- Polynomial regression use ordinary linear regression to indirectly fit data to polynomial expressions
- Vulnerable to noise

1. Logistic Regression
- Predicts the probability of an observation belongs to one of the two classes (binary classifer)
- Outputs probability
- Cost function or log-loss needs to be minimized
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib as plt
%matplotlib inline

df = pd.read_csv('link')

df_std = StandardScaler().fit_transform(df[['col1','col2']])

x_train, x_test, y_train, y_test = train_split_test(df_std,df['y-axis'],test_size=int,random_state=int)

LR = LogisticRegression().fit(x_train,y_train)
yhat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)

log_loss(y_test,yhat_prob)
```
