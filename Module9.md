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
- hierarchical clustering: divisive: one root cluster iteratively split into smaller child cluster; agglomerative: individual cluster merges into larger parents

### K-Means Clustering
- divides data into k non-overlapping clusters
- K-clusters have minimal variances around centroids and maximal dissimilarity between clusters
- assumes convex clusters and balanced cluster sizes
- steps:
1. Initialize the algorithm: select the value of k; randomly select k centroids
2. Iteratively assign points to cluster and update centroids: compute distance, assign each point to closest centroid; update centroid
3. Repeat until centroids stabilize or max iterations reached
- determining k:
1. Silhouette analysis: Measures cohesion and separation
2. Elbow method: Plot for different cluster numbers
3. Davies-Bouldin Index: Measures each cluster's average similarity ratio
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
%matplotlib inline

df = pd.read_csv('link')

# KMeans
k_mean = KMeans(init="kmeans++",n_clusters=int,n_init=int)
k_mean.fit(df)

# visualize
fig = plt.figure(figsize=(6,4))
colors = plt.cm.table(np.linspace(0,1,len(set(k_mean.labels_))))
ax = fig.add_subplot(1,1,1)
for k,col in zip(range(len([[int,int],[int,int],...)))
  cluster_member = (k_mean.labels_==k)
  cluster_center = k_mean.cluster_centers_[k]
  ax.plot(df[cluster_member,0],df[cluster_member,1],'w',markerfacecolor=col,marker='.',ms=int)
  ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
ax.set_title('K_Means')
ax.set_xticks(())
ax.set_yticks(())
plt.show()
```

### DBSCAN Clustering
- density-based spatial clustering algorithm
- create clusters centered around spatial centroids
- discovers clusters of any shape, size, or density (distinguish noise)
- every data set is labelled as: (1) core point (2) border point (3) noise point

### HDBSCAN Clustering
- locally adjusts neighborhood radii for cluster stability
- combination of agglomerative and density-based clustering
- steps:
1. identifying each point as its own cluster
2. progressively agglomerates clusters into a hierarchy
3. simplified into a condensed tree

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan

df = pd.read_csv('link')

# dbscan
min_sample = int
eps = float
dbscan = DBSCAN(eps=eps,min_samples=min_samples,metric='euclidean')
clusters = dbscan.fit_predict(df)

# hdbscan
min_samples = None
min_cluster_size = int
hdb = HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size,metric='euclidean')
clusters2 = hdb.fit_predict(df)
```

## Clustering, Dimension Reduction & Feature Engineering
- cluserting: helps with feature selection; supports dimension reduction; enhance computational efficiency & scalability
- dimension reduction: simplifies visualization of high-dimensional clustering; reduce the number of features required [PCA, t-SNE, UMAP]

### Dimension Reduction Algorithms
1. Principle Component Analysis (PCA)
- assumes dataset are linearly correlated
- transforms features into principle components and retain variance
- principle components: orthogonal to each other; define a new coordinate system; organized in decreasing order of importance; first few components contain most of the information
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('link')

# pca
pca = PCA(n_components=int)
x = pca.fit_transform(df)
```
2. T-distributed Stochastic Neighbor Embedding (t-SNE)
- maps high-dimensional dataset to a lower-dimensional space
- good at finding clusters in complex, high-dimensional data
- focuses on preserving similarity of points close together
- don't scale well and difficult to tune
3. Uniform Manifold Approximation and Projection (UMAP)
- constructs a high-dimensional graph representation of the data based on manifold theory
- scales better than t-SNE
- preserve the global structure of data
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as UMAP
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('link')

# t-SNE
tsne = TSNE(n_components=int,random_state=int,perplexity=int,max_iter=int)
x_tsne = tsne.fit_transform(df)

#UMAP
umap_model = UMAP.UMAP(n_components=int,random_state=int,min_dist=float,spread=int,n_jobs=int)
x_umap = umap_model.fit_transform(df)
```

## Classification Metrics and Evaluation Techniques
- train/test split technique is used to evaluate model performance
- accuracy: ratio of correctly predicted instances
- confusion matrix: breaks down ground truth instances of a class
- precision: measures how many predicted positive instances are positive
- recall: measures how many positive instances are correctly predicted
- F1-score: combines precision and recall to represent a model's accuracy
```
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

accuracy_score(y_test,yhat)
classification_report(y_test,yhat)
confusion_matrix(y_test,yhat)
```

## Regression Metrics and Evaluation Techniques
- mean absolute error (MAE): average absolute difference between y_test and predicted y_test
- mean squared error (MSE): sum of squared difference between y_test and predicted y_test
- root mean squared error (RMSE): sqaure root of the MSE
- R^2: amount of variance in the dependent variable that the independent variable can explain
```
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

mean_squared_error(y_test, yhat)
root_mean_squared_error(y_test, yhat)
mean_absolute_error(y_test, yhat)
r2_score(y_test, yhat)
```

## Unsupervised Model Evaluation: Heuristics and Techniques
- heuristics:
1. internal evaluation metrics: rely on input data
   - Silhouette score: compares cohesion with each cluster; ranges from -1 to 1; higher value indicate better-defined clusters
   - Davies-Bouldin Index: measures cluster compactness ratio; lower values indicate more distinct clusters
   - Inertia(K-Means): calculates sum of variance within clusters; lower value suggest more compact clusters
2. external evaluation metrics: use ground truth labels
   - Adjusted Rand Index: measures similarity between true labels and outcomes; ranges from -1 to 1 [score 1: perfect alignment; score 0: random clustering; score -1: worse than random performance]
   - Normalized Mutual Information: quantifies shared information between cluster assignments; ranges from 0 to 1 [score 1: perfect alignment; score 0: no shared information]
   - Fowlkes-Mallows Index: calculates geometric mean of precision and geometric mean of recall
3. generalizability or stability evaluation: assess cluster consistency
4. dimensionality reduction techniques: visualize clustering outcomes
5. cluster-assisted learning: refines clusters
6. domain expertise: provides feedback
```
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

silhoutte_score(x,labels)
silhoutte_samples(x,labels)
```

## Cross-Validation and Advanced Model Validation Techniques
- optimize model without jeopardizing its ability to predict well on unseen data
- prevent overfitting by selecting the best configuration by tuning hyperparameters
- cross-validation algorithm:
1. split dataset into testing data, training set and validation set
2. optimize the model by repeatedly training it on training set and measuring its performance on validation set
3. choose the best set of hyperparameters and evaluate the result on testing data
- k-fold cross-validation algorithm:
1. divide the data into k equal-sized folds to be used as validation subsets
2. for each trial train the model based on remaining k-1 folds
3. test the model on selected fold and store its model's score

## Regularization in Regression and Classification
- constrains model complexity by discouraging perfect fitting
- penalizes larger coefficients by reducing their magnitude
- Ridge and Lasso are regularized forms of linear regression that differ in their cost functions
```
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# ridge
ridge_reg = Ridge(alpha=int)
ridge_reg.fit(x,y)
yhat = ridge_reg.predict(x)

# lasso
lasso_reg = Lasso(alpha=float)
lasso_reg.fit(x,y)
yhat = lasso_reg.predict(x)
```

