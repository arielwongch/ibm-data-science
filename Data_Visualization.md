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

## Other Advanced Visualizations
1. Waffle charts
- represents categorical data in square tiles and cells
- displays proportion or percentage of different categories
```
import matplotlib.pyplot import plt
from pywaffle import Waffle

fig = plt.figure(FigureClass=Waffle,rows=<row>,columns=<col>,values=df,legend=legend)
```
2. Regression Plots
```
import seaborn as sns

sns.regplot(x='col1',y='col2',data=df)
```
3. Categorical Plots
```
import seaborn as sns

sns.countplot(x='col1',data=df)
```
4. Maps
- folium is a library primarily to visualize geospatial data
- folium uses latitude and longitude to create a map of any location
```
import folium

# default world map
map = folium.Map()

# worrld map centered at [latitude,longitude], zoom = int, map style = 'style'
map = folium.Map(location=[latitude,longitude],zoom_start=<int>, tiles='<style>')

# add markers to the map
folium.Marker(location=[latitude,longitude],popup='<style>').add_to(map)
```

## Dashboard
- Plotly: visualization can be displayed in Jupyter notebook; saved to HTML files; used in developing Python-built web applications
```
import plotly.graph_objects as go
import plotly.express as px

# basic plot
# method 1
fig = go.Figure(data=go.Scatter(x=x,y=y))
fig.update_layout(title='title',xaxis_title='xaxis',yaxis_title='yaxis')
fig.show()
# method 2
fig = px.line(x=x,y=y,title='title',labels=dict(x='xaxis',y='yaxis'))
fig.show()
```
- Dash: User Interface Python library from Plotly; easy to build GUI
```
import pandas as pd
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

df = pd.read_csv('link')

# create a dash object
app = dash.Dash()

# dash app layout
app.layout = html.Div(children=[ html.H1('title',style={'textAlign':'center','font-size':40}),
                                 html.Div(["Input: ",dcc.Input(id='input1',value='default_value1',type='<type>',style={'font-size':35}),],style={'font-size':40}),
                                 html.Br(),
                                 html.Br(),
                                 html.Div(dcc.Graph(id='plot1')),
                                 ])

# callback
# one input
@app.callback(
              Output(component_id='plot1',component_property='figure'),
              Input(component_id='input1',component_property='value')
             )
# two input
@app.callback(
              Output(component_id='plot1',component_property='figure'),
              [Input(component_id='input1',component_property='value'),
               Input(component_id='input2',component_property='value')]
             )

def get_graph('selected_input1'):
  # data preprocessing
  ...

  # plot the graph
  fig1 = px.<kind>(...)

  fig1.update_layout()
  return fig1

# main
if __name__ == '__main__':
  app.run()
```
