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

## Classification
- Supervised ML method, uses fully trained models to predict labels on new data
- labels form a categorical variable with discrete values
- common classification algorithm: naive bayes, logistic regression, decision trees, KNN, Support Vector Machines, NN
- Multiclass classifier: Logistic Regression, Decision Trees, KNN
- Strategies to entend binary classifiers to multiclass classifier: One-versus-all, One-versus-one

**One-Versus-All**
-use one binary classifier for each class label (binary prediction for every data point for a one-versus-the-rest classifier)

**One-Versus-One**
-use binary classifier on all possible pairs of classes
-the final class scheme is determined by a voting scheme (popularity, weighted vote)

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline

df = pd.read_csv('link')

# preprocess
# select and standardize continous numerical features 
continous_column = df.select_dtype(include=['float64']).columns.tolist()
continous_std = StandardScaler.fit_transform(df[continous_column])
# combine with original dataset
df_std = pd.DataFrame(continous_std,columns=StandardScaler.get_feature_names_out(continuous_column))
df_scaled = pd.concat([df.drop(columns=continous_column),df_std],axis=1)

# one-hot encoding
# select all categorical column (except target column)
categorical_column = df_scaled.select_dtypes(include=['object']).columns.tolist()
# apply one-hot encoding
encoder = OneHotEncoder(sparse_output=False,drop='first')
encoded_std = encoder.fit_transform(df_scaled[categorical_column])
# combine with original data
df_encoded = pd.DataFrame(encoded_std,columns=StandardScaler.get_feature_names_out(categorical_column))
df_prep = pd.concat([df.drop(columns=categorical_column),df_encoded],axis=1)

# one-hot encoding (target column)
df_prep['target'] = df_prep['target'].astype('category').cat.codes

# split dataset
x = df_prep.drop('target',axis=1)
y = df_prep['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=int,random_state=int)

# one vs one classifier
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(x_train,y_train)
yhat = model_ovo.predict(x_test)

accuracy_score(y_test,yhat)
```

## Decision Tree
