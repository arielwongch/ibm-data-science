# Machine Learning

## Simple and Logistic Regression
- models linear relationship between continuous target variable and explanatory features
- can be used to predict a continuous value
- a single independent variable estimates dependent variable
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

df = pd.read_csv('link')

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
- uses two or more independent variables to estimate a dependent variable
- MLR performs better than SLR, but too many variables can cause overfitting
- to improve prediction, convert categorical independent variables into numerical variables
```
import numpy as np
import matplotlib.pyplot as plt
import panadas as pd
%matplotlib inline

df = pd.read_csv('link')

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
- models non-linear relationship between dependent variable and independent variables
- polynomial regression use ordinary linear regression to indirectly fit data to polynomial expressions
- vulnerable to noise

1. Logistic Regression
- predicts the probability of an observation belongs to one of the two classes (binary classifer)
- outputs probability
- cost function or log-loss needs to be minimized
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
- supervised ML method, uses fully trained models to predict labels on new data
- labels form a categorical variable with discrete values
- common classification algorithm: naive bayes, logistic regression, decision trees, KNN, Support Vector Machines, NN
- multiclass classifier: Logistic Regression, Decision Trees, KNN
- strategies to entend binary classifiers to multiclass classifier: One-versus-all, One-versus-one

**One-Versus-All**
- use one binary classifier for each class label (binary prediction for every data point for a one-versus-the-rest classifier)

**One-Versus-One**
- use binary classifier on all possible pairs of classes
- the final class scheme is determined by a voting scheme (popularity, weighted vote)

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
- each internal node corresponds to a test; each branch corresponds to the result of the test; each leaf node assigns a class to a data
- pruning simplifies decision tree, pruned tree is more concise and higher accuracy
- decision tree select feature that best split the data to train the tree
- common split measures: information gain, gini impurity

**Entropy**
- measure of information disorder in a dataset (how random the classes in a node are)
- if classes are completely homogeneous, entropy=0; if classes are equally divided, entropy=1
- information gain: opposite of entropy

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
%matplotlib inline

df = pd.read_csv('link')

# preprocess
label_encoder = LabelEncoder()
df['numerical_col'] = label_encoder.fit_transform(df['numerical_column'])

# split dataset
x = df.drop('target',axis=1)
y = df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=int,random_state=int)

# decision tree
dtree = DecisionTreeClass(criterion='entropy',max_depth=int)
dtree.fit(x_train,y_train)
yhat = dtree.predict(x_test)
metrics.accuracy_score(y_test,yhat)

# plot decision tree
plot_tree(dtree)
plt.show()
```

## Regression Tree
- analogous to decision tree that predicts continuous values
- outputs average value of target values
- splitting criterion: features that minimize error between actual and predicted value (MSE)
- categorical feature split: one-vs-one / one-vs-all
- continuous feature split: sort all feature value; drop duplicates; define midpoint as threshold
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('link')

# preprocess and split
x = normalize(df.drop(['target'],axis=1).values,axis=1,norm='l1',copy=False)
y = df['target].values.astype('float64')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=int,random_state=int)

# regression tree
regressor = DecisionTreeRegressor(criterion='squared_error',max_depth=int,random_state=int)
regressor.fit(x_train,y_train)
yhat = regressor.predict(x_test)

mean_squared_error(x_test)
```

## Support Vector Machine (SVM)
- maps each data as a point in multidimensional space; input features are represented as a value for a specific coordinate
- create a hyperplane that segregates dataset
- tolerates soft margin and misclassifications while maximizing the margin
- parameter C controls the margin (smaller C: softer margin, more misclassification; larger C: harder margin, stricter separation)
- decision boundary is hyperplane that maximize margin
- for non-linear separatable data, we map data into a higher dimensional space (x,y)->(x,y,z), z=x2+y2
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
%matplotlib inline

df = pd.read_csv('link')

# preprocessing and split
df[['col1','col2',...]] = StandardScaler().fit_transform(df[['col1','col2',...]])
x = normalize(df[['col1','col2',...]],norm='l1')
y = df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=int,random_state=int)

# SVM
svm = LinearSVC(class_weight='balanced',random_state=int,loss='hinge',fit_intercept=False)
svm.fit(x_train,y_train)
yhat = svm.predict(x_test)

roc_soc_score(y_test,yhat)
```

## KNN
- pick a value for k -> for each data, compute the distance to labeled points ->select knn -> predict class (mean,median)
- effect of k: k too small: value fluctuate, overfitting; k too large: finer details lost, underfitting
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline

df = pd.read_csv('link')

# preprocessing and split
x = StandardScaler().fit_transform(df.drop('target',axis=1))
y = df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=int,random_state=int)

# KNN
k = int
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(x_train,y_train)
yhat = knn_model.predict(x_test)

accuracy_score(y_test,yhat)
```

## Bias, Variance, Ensemble Models
- bias measures the accuracy of predictions (average diff between yhat and y_test)
- variance measures prediction fluctuations (high variance: sensitive to training data)
- ensemble learning: bagging & boosting
- decision tree serves as base learner, tree depth adjusts bias and variance
- averaging the models reduces prediction variance and lowers the risk of overfitting

**Random Forest**
- use bagging for training, train decision tree on bootstrapped dataset
- focus on minimizing prediction bias

**Boosting**
- builds a series of weak learners, each learner corrects the previous learner's error
- mitigate overfitting

**Bagging**
- increase variance, mitigate underfitting
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv('link')

# preprocessing and split
x = df[['col1','col2',...]]
y = df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=int,random_state=int)

# Random Forest and xgb boosting
n_estimators = int
# random forest
rf = RandomForestRegressor(n_estimators=n_estimators,random_state=int)
rf.fit(x_train,y_train)
rf_yhat = rf.predict(x_test)
# xgb
xgb = XGBRegressor(n_estimators=n_estimators,random_state=int)
xgb.fit(x_train,y_train)
xgb_yhat = xgb.predict(x_test)

mean_squared_error(y_test,rf_yhat)
mean_squared_error(y_test,xgb_yhat)
r2_score(y_test,rf_yhat)
r2_score(y_test,xgb_yhat)
```

## Clustering
