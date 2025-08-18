# Database

## Introduction to Database
- Structured Query Language (SQL): used for relational database to query data out of a database
- Database: a repository of data; provides the functionality for adding, modifying, and querying data
- DataBase Management System (DBMS): a set of software tools for the data in database
- Relational DataBase Management System (RDBMS): a set of software tools that controls the data

**Relational Model**
- allow for data independence; data stored in tables
- entities become tables; attributes get translated into columns

**SQL Statements**
1. Data Definition Language (DDL) statements
- define, change, or drop data
2. Data Manipulation Language (DML) statements
- read and modify data
- often refers to CRUD operations (Create, Read, Update, Delete)

## SQL 
- main purpose of a DBMS is to store and facilitate retrieval of data
- SELECT statement
```
# select all column
SELECT * FROM <table>

# select some columns
SELECT <column1>,... FROM <table>

# 'where' clause to restrict the result set
SELECT * FROM <table> WHERE <condition>

# select all column with column = 'value' (compare: = , < , > , <= , >= , <>)
SELECT * FROM <table> WHERE <column1> = 'value1'

# 'count' clause to count the number of rows
SELECT COUNT(<column1>) FROM <table>

# 'distinct' to get unique values
SELECT DISTINCT <column1> FROM <table>

# 'limit' to limit the number of rows selected
SELECT * FROM <table> LIMIT <int>

# select all columns with column in a set of values
SELECT * FROM <table> WHERE <column1> IN ('<value1>','<value2>')

# select all column with column starting/containing/ending string pattern
SELECT * FROM <table> WHERE <column1> LIKE '(%)<string>(%)'

# select all column with column within a range (inclusive)
SELECT * FROM <table> WHERE <column1> BETWEEN <value1> AND <value2>

# select all column ordered by column1 (ascending order by default)
SELECT * FROM <table> ORDER BY <column1>

# select all column ordered by column1 (descending order)
SELECT * FROM <table> ORDER BY <column1> DESC

# group result group by column1
SELECT <column1>, COUNT(<column1>) AS <column> FROM <table> GROUP BY <column1>

# restrict the group result
SELECT <column1>, COUNT(<column1>) AS <column> FROM <table> GROUP BY <column1> HAVING <condition>
```
- other Data Manipulation Language (DML) statement
```
# insert a new row into the table
INSERT INTO <table> (<column1>,...) VALUES ('<value1>',...)

# insert multiple rows into the table
INSERT INTO <table> (<column1>,...) VALUES
('<value11>',...), ('<value21>',...)

# update values for specific rows in the table
UPDATE <table> SET <column1> = <value1>

# delete a row from a table
DELETE FROM <table> WHERE <condition>
```
- Data Definition Language (DDL) statements
```
# create a new table
CREATE TABLE <table>
  (
  <column1> <datatype> <optional paramters>,
  ...
  )
# optional paramters
# 1. PRIMARY KEY 2. NOT NULL

# add or remove column
ALTER TABLE <table>
  ADD COLUMN <column1> <datatype>
  MODIFY <column1> <datatype>
  DROP COLUMN <column1>

# delete a table
DROP TABLE <table>

# delete data from a table
TRUNCATE TABLE <table> IMMEDIATE
```
