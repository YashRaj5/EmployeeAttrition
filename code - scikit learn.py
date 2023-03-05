# Databricks notebook source
# MAGIC %md
# MAGIC TASK :- To predict whether an employee will choose attrition or not.
# MAGIC 
# MAGIC To perform such task where we have to predict an event we are going to use Machine Learning as the programming paradigm. In which with the help of mathematical model, and huge amount of data, machine learns the pattern and create it's own logic and then with the learned logic, it gives the ouput on unseen data.
# MAGIC 
# MAGIC The task here is to classify whether an employee will under go attrition or not. For that we are going to be using Decision Tree algorithm first, and then according to the requirement we may choose another algorithm. More info will be provided later.

# COMMAND ----------

# MAGIC %md
# MAGIC * PROCESS
# MAGIC - Problem Fomulation : Meaning, what we are trying to achieve with the help of technology
# MAGIC - Gathering of raw data : Data on from which inference is to be deduced.
# MAGIC - Data Processing : Data which is collected in previous stage is needed to be made useable. It may contain null values, or redundant values, categorical values. This is needed to be done in order to make the whole data understandable by the system.
# MAGIC - Splitting the raw data : For our algorithm ot write it's own rules, it needs to calculate it's parameters and for than a sample of the data referred to as train data, is provided and processed and then the performance is tested with the help of another set of data called as test data.
# MAGIC - Model evaluation : Performance is measured and then the model is deployed in case of production environment.

# COMMAND ----------

# MAGIC %md
# MAGIC Reading the Data

# COMMAND ----------

# importing all the pyspark functions
from pyspark.sql.functions import *

# COMMAND ----------

df_main = spark.read.csv("dbfs:/FileStore/WA_Fn_UseC__HR_Employee_Attrition.csv", header=True, inferSchema=True)

# COMMAND ----------

df_main.display()

# COMMAND ----------

df_main.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the machine learning implementation, some columns will be removed as they won't be helpful in the prediction

# COMMAND ----------

df_main = df_main.select(col("Age"), col("Attrition"), col("BusinessTravel"), col("Department"), col("DistanceFromHome"), col("EducationField"), col("EnvironmentSatisfaction"), col("Gender"), col("JobLevel"), col("JobRole"), col("JobSatisfaction"), col("MaritalStatus"), col("MonthlyIncome"), col("NumCompaniesWorked"), col("PercentSalaryHike"), col("RelationshipSatisfaction"), col("TotalWorkingYears"), col("TrainingTimesLastYear"), col("YearsAtCompany"), col("YearsInCurrentRole"), col("YearsSinceLastPromotion"), col("YearsWithCurrManager"))

# COMMAND ----------

df_main = df_main.withColumn('Attrition', when(df_main.Attrition == 'No', 0).otherwise(1))

# COMMAND ----------

display(df_main)

# COMMAND ----------

# MAGIC %md 
# MAGIC Due to lack of data, the given data will be divided for training and testing purpose and then for implimenting the model on unseen data.

# COMMAND ----------

# Defining the number of splits
n_splits = 2

# Calculate the count of each dataframe rows
each_len = df_main.count() // n_splits

# Creating a copy of orignal data frame
copy_df = df_main

# Iterating for each dataframe
i=1
while i < n_splits:
    # Get the top 'each_len' number of rows
    temp_df = copy_df.limit(each_len)
    # Turncate the 'copy_df' to remove the contents fetched for 'temp_df'
    copy_df = copy_df.subtract(temp_df)
    # View the dataframe
    temp_df.show(truncate=False)
    # Increment the split values
    i+=1

# COMMAND ----------

display(copy_df)
display(temp_df)

# COMMAND ----------

# Now we have two sets of data, one 'copy_df' which is going to be used for training and testing purpose and then we have 'temp_df' which is going to be use for show real world implementating of model
df_0 = copy_df
unseen_df = temp_df.drop("Attrition")

# COMMAND ----------

# MAGIC %md
# MAGIC As number of rows present here are not much, so we are going to be using pandas API

# COMMAND ----------

raw_data = df_0.toPandas ()
display(raw_data)

# COMMAND ----------

# dimentions of the raw data
print(raw_data.shape)
# first 5 rows of the data
raw_data.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Importing Required Libraries

# COMMAND ----------

# Packages / libraries
import os #provides functions for interacting with the operating system
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

%matplotlib inline

# To install sklearn type "pip install numpy scipy scikit-learn" to the anaconda terminal

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

# Datetime lib
from pandas import to_datetime
import itertools
import warnings
import datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

# COMMAND ----------

# MAGIC %md
# MAGIC Data Pre-Processing

# COMMAND ----------

# Investigate all the elements whithin each Feature 

for column in raw_data:
    unique_vals = np.unique(raw_data[column])
    nr_values = len(unique_vals)
    if nr_values < 12:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))

# COMMAND ----------

# Checking for null values
raw_data.isnull().sum()

# COMMAND ----------

raw_data.columns

# COMMAND ----------

# Limiting the data
raw_data2 = raw_data[['Age', 'Attrition', 'Gender', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]

# Visualize the data using seaborn Pairplots
g = sns.pairplot(raw_data2, hue = 'Attrition', diag_kws={'bw': 0.2})

# COMMAND ----------

# Investigate all the features by our y

features = ['Age', 'Gender', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']


for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=raw_data2, hue = 'Attrition', palette="Set1")

# COMMAND ----------

raw_data2.head()

# COMMAND ----------

# Making categorical variables into numeric representation

new_raw_data = pd.get_dummies(raw_data2, columns = ['MaritalStatus', 'Gender'])
new_raw_data.head()

# COMMAND ----------

new_raw_data = new_raw_data.replace(to_replace = ['Yes','No'],value = ['1','0']) 

# COMMAND ----------

# Scaling our columns

scale_vars = ['MonthlyIncome']
scaler = MinMaxScaler()
new_raw_data[scale_vars] = scaler.fit_transform(new_raw_data[scale_vars])
new_raw_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting of raw data

# COMMAND ----------

X = new_raw_data.drop('Attrition', axis=1).values# Input features (attributes)
y = new_raw_data['Attrition'].values # Target vector
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size=0.3, random_state=0)

# COMMAND ----------

# MAGIC %md
# MAGIC Running Descision Tree
# MAGIC Decision Trees are a supervised learning method used for classification and regression
# MAGIC How it works:
# MAGIC 
# MAGIC The ID3 algorithm begins with the original set {S} S as the root node
# MAGIC On each iteration of the algorithm, it iterates through every unused attribute of the set and calculates the entropy (or information gain) of that attribute
# MAGIC It then selects the attribute which has the smallest entropy (or largest information gain) value.
# MAGIC The set is then split by the selected attribute to produce subsets of the data.
# MAGIC The algorithm continues to recurse on each subset, considering only attributes never selected before.

# COMMAND ----------

dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
dt.fit(X_train, y_train)

# COMMAND ----------

!pip install graphviz

# COMMAND ----------

import graphviz 

dot_data = tree.export_graphviz(dt, out_file=None, 
    feature_names=new_raw_data.drop('Attrition', axis=1).columns,    
    class_names=new_raw_data['Attrition'].unique().astype(str),  
    filled=True, rounded=True,  
    special_characters=True)
graph = graphviz.Source(dot_data)

# COMMAND ----------

graph

# COMMAND ----------

#del final_fi

# Calculating FI
for i, column in enumerate(new_raw_data.drop('Attrition', axis=1)):
    print('Importance of feature {}:, {:.3f}'.format(column, dt.feature_importances_[i]))
    
    fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [dt.feature_importances_[i]]})
    
    try:
        final_fi = pd.concat([final_fi,fi], ignore_index = True)
    except:
        final_fi = fi
        
        
# Ordering the data
final_fi = final_fi.sort_values('Feature Importance Score', ascending = False).reset_index()            
final_fi

# COMMAND ----------

# Accuracy on Train
print("Training Accuracy is: ", dt.score(X_train, y_train))

# Accuracy on Train
print("Testing Accuracy is: ", dt.score(X_test, y_test))

# COMMAND ----------

# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# COMMAND ----------

y_pred = dt.predict(X_train)

# Plotting Confusion Matrix
cm = confusion_matrix(y_train, y_pred)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=dt.classes_, title='Training confusion')

# COMMAND ----------

y_pred = dt.predict(X_train)
y_pred
confusion_matrix(y_train, y_pred)

# COMMAND ----------

# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)
FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print ("The True Positive rate / Recall per class is: ",TPR)

# Precision or positive predictive value
PPV = TP/(TP+FP)
print ("The Precision per class is: ",PPV)

# False positive rate or False alarm rate
FPR = FP/(FP+TN)
print ("The False Alarm rate per class is: ",FPR)

# False negative rate or Miss Rate
FNR = FN/(TP+FN)
print ("The Miss Rate rate per class is: ",FNR)

# Classification error
CER = (FP+FN)/(TP+FP+FN+TN)
print ("The Classification error of each class is", CER)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print ("The Accuracy of each class is", ACC)
print("")

##Total averages :
print ("The average Recall is: ",TPR.sum()/2)
print ("The average Precision is: ",PPV.sum()/2)
print ("The average False Alarm is: ",FPR.sum()/2)
print ("The average Miss Rate rate is: ",FNR.sum()/2)
print ("The average Classification error is", CER.sum()/2)
print ("The average Accuracy is", ACC.sum()/2)

# COMMAND ----------

# MAGIC %md
# MAGIC Deploying our model on NEW, unseen data

# COMMAND ----------

unseen_df_pd = unseen_df.toPandas()
# print the shape
print(unseen_df_pd.shape)

# Viz
unseen_df_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Performing same pre-processing

# COMMAND ----------

# Limiting the data
unseen_df_pd2 = unseen_df_pd[['Age', 'Gender', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]

# COMMAND ----------

unseen_df_pd2.head()

# COMMAND ----------

# Making categorical variables into numeric representation

new_unseen_df_pd = pd.get_dummies(unseen_df_pd2, columns = ['MaritalStatus', 'Gender'])
new_unseen_df_pd.head()

# COMMAND ----------

# Scaling our columns

scale_vars = ['MonthlyIncome']
scaler = MinMaxScaler()
new_unseen_df_pd[scale_vars] = scaler.fit_transform(new_unseen_df_pd[scale_vars])
new_unseen_df_pd.head()

# COMMAND ----------

# Making Pridictions
pred_decitree = dt.predict(new_unseen_df_pd.values)
pred_prob_decitree = dt.predict_proba(new_unseen_df_pd.values)

pred_decitree

# COMMAND ----------

pred_prob_decitree

# COMMAND ----------

# function to select second column for probabilities
def column(matrix, i):
    return [row[i] for row in matrix]

column(pred_prob_decitree, 1)

# COMMAND ----------

# Joining the raw data witht the predictions

output = unseen_df_pd.copy()
output['Predictions - Attrition or Not'] = pred_decitree
output['Predictions - Probability to Attitional'] = column(pred_prob_decitree, 1)
output['Predictions - Attrition or Not Desc'] = 'Empty'
output['Predictions - Attrition or Not Desc'][output['Predictions - Attrition or Not'] == 0] = 'Retention'
output['Predictions - Attrition or Not Desc'][output['Predictions - Attrition or Not'] == 1] = 'Attrition'
output.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Saving output

# COMMAND ----------

os.getcwd()

# COMMAND ----------

output.to_csv("/databricks/driver/FileStore/Attrition Predictions Output")

# COMMAND ----------

output_df = spark.createDataFrame(output)
display(output_df)

# COMMAND ----------

output_df.write.csv("dbfs:/FileStore/Attrition Predictions Output")

# COMMAND ----------

# renaming columns according to sql standards
output_df = output_df.withColumnRenamed('Predictions - Attrition or Not Desc', 'Attrition_or_Not_Desc')

# COMMAND ----------

output_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Connecting to SQL

# COMMAND ----------

jdbcHostname = "databaseyr.database.windows.net"
jdbcPort = 1433
jdbcDatabase = "db_training"
jdbcUsername = "trianee_YR"
jdbcPassword = "Pa$$word1234"
 
jdbcUrl = f"jdbc:sqlserver://{jdbcHostname}:{jdbcPort};databaseName={jdbcDatabase};user={jdbcUsername};password={jdbcPassword}"

# COMMAND ----------

output_df.write.format("jdbc") \
  .mode("overwrite") \
  .option("url", jdbcUrl) \
  .option("dbtable", "attrition_output") \
  .option("user", jdbcUsername) \
  .option("password", jdbcPassword) \
  .save()
