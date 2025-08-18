# Data Extraction

## Open Function
- mode: "r": reading; "w": writing; "a": appending
```
# create a file object
file1 = open("/path/textfile.txt","r")

# file attributes
file1.name
file1.mode

# close the file object
file1.close()
```
- open file with 'with' statement (automatically close the file)
```
with open("/path/textfile.txt","r") as file1:
  file_content = file1.read()
```
- write the file with mode "w"
```
with open("/path/textfile.txt","w") as file1:
  file1.write('content')
```

## Pandas
- read comma separated values (csv)
```
import pandas as pd

# read csv file
df = pd.read_csv('link')

# dataframe attributes
df.head()    # first 5 rows
```
- create a DataFrame object
```
# create a dictionary
dict = {  'col1': ['value11','value12',...]
          'col2': ['value21','value22',...]
          ...
        }

# create a df object
df = pd.DataFrame(dict)
```
- get a whole column
```
# get one column
x = df[['col1']]

# get multiple column
y = df[['col1','col2']]
```
- access elements
```
# accessing the element in xth row, yth column
df.iloc[x,y]

# accessing the element in xth row at 'col1'
df.iloc[x,'col1']

# accessing a range of elements
df.iloc[a:b,c:d]

# slicing
df.loc[x,y]
```
- working with data
```
# get unique element in column
df['col1'].unique()

# get elements with conditions
df1 = df[df['col1']<condition>]
```
- save as csv file
```
df.to_csv('name')
```

## NumPy
- creata a numpy object
```
import numpy as np

# create a numpy object
arr = np.array([list])

# numpy array attibtes
arr[i]      # accessing the ith element in the array
arr.dtype   # get the datatype of the elements in the array
arr.size    # number of elements in the array
arr.ndim    # number of dimension of the array
arr.shape   # dimension of the array
```
- numpy functions
```
np.dot(x,y)
np.linspace(start,end,#interval)
```

## API & Data Collection
### API
- API allow communications between two software
- REST API: REpresentational State Transfer API
- REST API are used to interact with web services
- http methods are a way of transmitting data over the internet
- REST API send a request via an HTTP message, which usually contains a JSON file
### URL
- Uniform Resource Locator (url) is the most popular way to find resources on the web
- URL:
1. Scheme: the protocol (http://)
2. Internet Address / Base URL: the location (www.github.com)
3. Route: location on web server (/arielwongch)
### Request Message
1. Start line: /GET/index.html HTTP/1.0
2. Header: additional information
3. Body: html file
### Response Message
1. Start Line: HTTP/1.0 <STATUS_CODE> OK
2. Header: additional information
3. Body: requested file
```
<!DOCTYPE html>
<html>
<body>
<h1>Header</h1>
<p>Content</p>
</body>
</html>
```
- Status Code
| Status Code | Description |
| ---- | ---- |
| 1XX | Informational |
| 100 | Everything So Far Is OK |
| 2XX | Success |
| 200 | OK |
| 3XX | Redirection |
| 300 | Mutlitple Choices |
| 4XX | Client Error |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 5XX | Server Error |
| 501 | Not Implemented |
- HTTP Method
| HTTP Method | Description |
| ---- | ---- |
| GET | Retrieves data from server |
| POST | Submits data to server |
| PUT | Updates data already on server |
| DELETE | Deletes data from server |
### Python Request
- GET Request
```
import requests

# get requests object
response = requests.get('link')

# requests attributes
response.status_code
response.request.headers
response.request.body
response.headers
response.headers['date']
response.headers.text
```
- GET Request with query string
```
import requests

url = 'link/get'
query_parameter = {"para1":"value1",...}

# get the response
response = requests.get(url,params=query_parameter)

# important attributes
response.json()
```

## Web Scrapping 
```
from bs4 import BeautifulSoup
import requests

# create a requests object
response = requests.get('link')

# create a soup object
soup = BeautifulSoup(response.text,'html.parser')

# find all the tag in the HTML
data_all = soup.find_all('<tag>')

data = []

# loop thr the list and append
for data in data_all:
  row = data.find_all('<tag2>')
  data.append(row.contents[0])
```

## Read Different File Formats
- JSON
```
import json

with open('file.json','r') as file1:
  obj = json.load(file1)
```
- XML
```
import pandas as pd
import xml.etree.ElementTree as etree

tree = etree.parse('file.xml')
root = tree.getroot()

columns = ['col1','col2',...]
df = pd.DataFrame(columns=columns)

for node in root:
  'col1' = node.find('col1').text
  ...
  df = df.append(pd.Series(['col1','col2',...]),ignore_index=True)
```










