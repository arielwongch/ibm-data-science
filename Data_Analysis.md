# Data Analysis

## Data Wrangling
1. Missing Value
- missing values occur when no data is stored for a feature in an observation
- could be represented as '?', 'N/A', '0' or just a blank cell
```
# check how many rows contain empty cell
df.isnull()

# drop missing values
df.dropna(subset=['col1'],axis=0.inplace=True)

# replace missing value
df['col1'].replace(np.nan,<new_value>)
```
2. Data Formatting
```
# new calculation
df['col1'] = df['col1'] <calculation>
df.rename(columns={'col1':'new_col1'},inplace=True)

# identify datatype
df.dtypes()

# correct datatype
df['col1'] = df['col1'].astype('<datatype>')
```
3. Data Normalization
- we should normalize the variables so that the range of the values is consistent
```
# Method 1: Simple Feature Scaling
# Xnew = Xold/Xmax
df['col1'] = df['col1'] / df['col1'].max()

# Method 2: Min-Max
# Xnew = (Xold-Xmin)/(Xmax-Xmin)
df['col1'] = (df['col1'] - df['col1'].min()) / (df['col1'].max() - df['col1'].min())

# Method 3: Z-score
# Xnew = (Xold-mean)/sd
df['col1'] = (df['col1'] - df['col1'].mean()) / df['col1'].std()
```
4. Binning
- converts numeric into categorical variables
```
bins = np.linspace(df['col1'].min(),df['col1'].max(),<#interval+1>)
group_names = ['name1',;name2',...]
df['col1'] = pd.cut(df['col1'],bins,labels=group_names,include_lowest=True)
```
5. One Hot Encode
- convert categorical into numeric variables
```
pd.get_dummies(df['col1'])
```

## Exploratory Data Analysis
- summarize main characteristics of data; uncover relationships between variables
- descriptive statistics
```
# summarize statistics
df.describe()

# summarize categorical data
df['col1'].value_counts()

# box plot
sns.boxplot(x='col1',y='col2',data=df)

# scatter plot
plt.scatter(df['col1'],df['col2'])

# group by categorical variables
df_grp = df[['col1','col2',...]]
df_grped = df_grp.groupby(['col1','col2'],as_index=False).mean()

# pivot tables
df_pivot = df_grp(index='x-axis',columns='y_axis')

# heatmap
plt.pcolor(df_pivot,cmap='RdBu')
plt.colorbar()
```
- correlation
```
# correlation between two features
sns.regplot(x='col1',y='col2',data=df)
plt.ylim(0,)
```

### Pearson Correlation
- measures the strength of correlation between two features
- correlation coefficient: close to +1: large positive relationship; close to -1: large negative relationship; close to 0: no relationship
- p-value: <0.001: strongly certainty in the result; <0.05: moderate certainty in the result; <0.1: weak certainty in the result; >0.1: no certainty in the result
```
import stats

pearson_coef_, p_value = stats.pearsonr(df['col1'],df['col2'])
```

## Model Development
- a model can be thought of as a mathematical equation used to predict a value given one or more other values

### Simple Linear Regression
- linear regression will refer to one independent variable to make a prediction
```
from sklearn.linear_model import LinearRegression

# create a linear regression object
lm = LinearRegression()

# fit the object with data
lm.fit(x,y)

# prediction
lm.predict(x_test)

```
### Polynomial Regression
- useful for describing curvilinear relationship
```
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=<int>,include_bias=False)

pr.fit_transform([<list>],include_bias=False)
```
```
from sklearn.preprocessing import StandardScaler

df_new = StandardScaler().fit_transform(df)
```
### Pipeline
```
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

input = [
        ('polynomial',PolynomialFeature(degree=<int>)),
        ('scale',StandardScaler()),)
        ('model',LinearRegression())
        ]
pipe = Pipeline(input)

pipe.fit(df[['col1','col2',...]],y)
```

## Model Evaluation and Refinement
- in-sample evaluation only tells us how well our model fit the training data

### train_test_split
- split dataset into random train and test subset
```
from sklearn.model_selection import train_test_split

# split dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=<float>,random_state=<int>)
```

### cross validation
```
from sklearn.model_selection import cross_val_score

scores = cross_val_score(<model obj>,x,y,cv=<#sections>)
```
```
from sklearn.model_selection import cross_val_predict

# predict
yhat = cross_val_predict(<model obj>,x,y,cv=<#sections>)
```

### ridge regression
```
from sklearn.model_selection import Ridge

# create a ridge object and predict
rm = Ridge(alpha=<float>)
rm.fit(x,y)
yhat = rm.predict(x)
```

### grid search
- allow us to scan through multiple free parameters, automatically iterating over hyperparameters using cross-validation
```
from sklearn.model_selection import GridSearchCV

# set a list of parameters
paramters = [{'alpha':[0.1,1,10]}]

grid = GridSearchCV(<model>,paramters,cv=<int>)
grid.fit(x,y)
```
