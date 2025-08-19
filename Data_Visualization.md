# Data Visualization

## Matplotlib
- basic plotting
```
import matplotlib.pyplot as plt
%matplotlib inline
```
- plot
```
# line plot
df.plot(kind='line')
plt.plot(x,y)

# area plot
df.plot(kind='area')
plt.fill_between(x,y)

# histogram
df.plot(kind='hist')
plt.hist(x)

# bar chart (vertical)
df.plot(kind='bar')
plt.bar(x,y)

# bar chart (horizontal)
df.plot(kind='barh')
plt.barh(x,y)

# pie chart
df.plot(kind='pie')
plt.pie(x)

# box plot
df.plot(kind='box')

# scatter plot
df.plot(kind='scatter')
plt.scatter(x,y)

# draw multiple plots
# method 1
fig, axs = plt.subplots(<row>,<col>)
axs[0].<kind>(x,y)
# method 2
axs1 = fig.add_subplot(<row>,<col>,<index_of_plot>)
axs1.<kind>(x,y)

# add/modify attributes
plt.title('title')
plt.xlabel('xaxis')
plt.ylabel('yaxis')
```
