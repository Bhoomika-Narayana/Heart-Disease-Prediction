#collection of data by importing libraries
import pandas

df=pandas.read_csv('heart.csv')
print(df)

#To print top 5 rows
print(df.head(5))

#To print bottom 5 rows
print(df.tail(5))

#To print the columns of data
print(df.columns)

#To check number of patients in data
print("No of patients in original data:"+str(len(df.index)))

#To check no of columns in data
x=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
print(x)
print("No of features in data:",+len(x.columns))

#To check no of targets in y variable
y=df['target']
print(y)

#To print the information about datatype
print(df.info())

#counting null values on each column
print(df.isnull().sum())

#count
print(df.describe())

#To know the sex of the patient
print(df["sex"].value_counts())