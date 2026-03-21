import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

df = pd.read_csv("studentperformance/student_data.csv")
print(df.head())

print("Shape of the Dataset: ",df.shape)
df.info()
df.describe()

#Checking the null values
print("Null values in dataset: ")
print(df.isnull().sum())

#Checking the duplicates Values in the dataset
print("Duplicates in the dataset: ")
print(df.duplicated().sum())

#Datatypes for each column in the dataset
print("Data types in the dataset: ")
print(df.dtypes)

#Finding the Average score for G1,G2,G3
df["AverageScore"] = (df["G1"]+ df["G2"]+df["G3"]) / 3
print("Average scores for G1, G2, G3:")
print(df["AverageScore"].head())

#Check the Highest Average Score in exam
highest_avg_score = df["AverageScore"].max()
print("Highest Average Score in the Exam: ",highest_avg_score)

#Graph Representation for G3 scores
plt.hist(df["G3"], bins = 10, color="blue")
plt.title("Distribution of G3 Scores")
plt.grid()
plt.xlabel("G3 Scores")
plt.ylabel("Frequency")
plt.show()

#Add Seaborn Library for Enhanced Graph representation
sns.histplot(df["G3"], bins = 10, color = "blue")
plt.title("Distribution of G3 scores")
plt.show()

#Compare the Average fimal grade (G3) between male and female students
print("Average Final Grade: ")
print(df.groupby("sex")["G3"].mean())

#Graph representation for avarage G3 by Gender
plt.bar(df.groupby("sex")["G3"].mean().index,df.groupby("sex")["G3"].mean().values,color=["red","green"])
plt.title("Average Final Grade: ")
plt.grid()
plt.xlabel("Gender")
plt.ylabel("Average G3 Score")
plt.show()

#Using the seaborn Library for the bar plot
sns.barplot(x="sex",y="G3",data=df)
plt.title("Average Final Grade")
plt.xlabel("Gender")
plt.ylabel("Average G3 Score")
plt.show()

#Find the Average G3 based on Medu and Fedu
print("Average based on Medu and Fedu: ")
print(df.groupby(["Medu","Fedu"])["G3"].mean())


#How does study time affect the final grades
print("Average G3 based on the study time: ")
print(df.groupby("studytime")["G3"].mean())

#Does Gooing out affect the grades
print("Average G3 based on the Going out: ")
print(df.groupby("goout")["G3"].mean())

#Alchol consumptio impact performance
print("Average G3 based on the Alchol Consumption: ")
print(df.groupby("Dalc")["G3"].mean())

#Relationship between absences and final grades
plt.scatter(df["absences"],df["G3"],color="pink")
plt.title("Relationship between Absences and Final Grades")
plt.grid()
plt.xlabel("Absences")
plt.ylabel("Final Grades")
plt.show()

#Using the Seaborn Library for the Scatter Plot
sns.scatterplot(x="absences",y="G3",data=df,color = "red")
plt.title("Relationship between Absence and Final Grades")
plt.xlabel("Absences")
plt.ylabel("Final Grades")
plt.grid()
plt.show()


#Student with more absence perfom worse than fewer absences
print("Average  G3 based on Absences:  ")
print(df.groupby("absences")["G3"].mean())

#Find the correlation between the G1,G2,G3
correlation = df[["G1","G2","G3"]].corr()
print("Correaltion Between G1,G2,G3: ")
print(correlation)

#which Feature is most correlated with G3
corr_G3 = correlation["G3"].drop("G3")
most_corr = corr_G3.abs().idxmax()
print("Most Correlated Feature : ",most_corr)

#Creating a New Column got Average Grade
df["Average Grade"] = (df["G1"] + df["G2"] + df["G3"]) / 3
print("New Column with Average Grade: ")
print(df["Average Grade"].head())

#Checkning the performs better in student with internet or student with romantic relationships
print("Average G3 based on Internet and Romantic Relationships: ")
print(df.groupby(["internet","romantic"])["G3"].mean())

#Predicting the Final Grades based on the other feature using the Linear Regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Input features and output/target variable
x = df[["Medu","Fedu","studytime","goout","absences"]]
y = df["G3"]

#Spliting the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Creating the Linear Regression model
model = LinearRegression()
model.fit(x_train,y_train)

#Making the Predictions
y_pred = model.predict(x_test)
print("Predicting Values: ",y_pred)

#Checking the Mean Squared Error and R2 Score of the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 Score: ",r2)

#Graph Representation of Actual vs Predicted Values
plt.scatter(y_test,y_pred,color="orange")
plt.xlabel("Actual G3 Scores")
plt.ylabel("Predicted G3 Scores")
plt.title("Actual vs Predicted G3 Scores")
plt.show()













