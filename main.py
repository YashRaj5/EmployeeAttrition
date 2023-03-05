# Databricks notebook source
# MAGIC %md
# MAGIC # Mounting data from Azure Data Lake

# COMMAND ----------

# mounting an ADLS 
dbutils.fs.mount(
source = "wasbs://raw@strgyr.blob.core.windows.net", # name of the storage account and the contianer 
mount_point = "/mnt/", # location of mounted storage
extra_configs = {"fs.azure.account.key.strgyr.blob.core.windows.net":"mml1UxXLJxrptnwDEIZGhQUd3HjS22ZBkKu/pIYau4GH2DjJ0+rEx44OyuHbRGys6Kq7qzevQFKZ+AStyzYu1g=="}) # key for connection to the storage

# COMMAND ----------

# listing files in the mounted storage
dbutils.fs.ls("/mnt/")

# COMMAND ----------

df=spark.read \
.option("header","True") \
.option("inferSchema","True") \
.option("sep",",") \
.csv("dbfs:/mnt/raw_data.csv")
print("There are",df.count(),"rows",len(df.columns),"columns","in the data.")

# COMMAND ----------

df.display()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().toPandas().transpose()

# COMMAND ----------

df.groupby("Attrition").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Checking for null values

# COMMAND ----------

from pyspark.sql.functions import count, when, isnan
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC Creating Age group

# COMMAND ----------

def udf_multiple(age):
    if (age<=25):
        return 'Under 25'
    elif (age >= 25 and age <= 35):
        return 'Between 25 and 35'
    elif (age > 35 and age < 50):
        return 'Between 36 and 49'
    elif (age >= 50):
        return 'Over 50'
    else: return 'N/A'

education_udf = udf(udf_multiple)
df=df.withColumn("AgeGroup", education_udf("Age"))
df=df.drop("Age")

# COMMAND ----------

# MAGIC %md
# MAGIC Creating groups for number of years workign in the company

# COMMAND ----------

def udf_multiple_yatc(yatc):
    if (yatc<=7):
        return 'Under 5'
    elif (yatc >= 7 and yatc <= 14):
        return 'Between 7 and 14'
    elif (yatc > 15 and yatc <= 22):
        return 'Between 15 and 22'
    elif (yatc > 23 and yatc <= 30):
        return 'Between 23 and 30'
    elif (yatc >= 31):
        return 'Over 30'
    else: return 'N/A'

education_udf = udf(udf_multiple_yatc)
df=df.withColumn("YearsAtCompanyGroup", education_udf("YearsAtCompany"))
df=df.drop("YearsAtCompany")

# COMMAND ----------

def udf_multiple_twy(twy):
    if (twy<=5):
        return 'Under 5'
    elif (twy >= 6 and twy <= 15):
        return 'Between 6 and 15'
    elif (twy > 16 and twy <= 25):
        return 'Between 16 and 25'
    elif (twy > 26 and twy <= 35):
        return 'Between 26 and 35'
    elif (twy >= 36):
        return 'Over 36'
    else: return 'N/A'

education_udf = udf(udf_multiple_twy)
df=df.withColumn("TotalWorkingYearsGroup", education_udf("TotalWorkingYears"))
df=df.drop("TotalWorkingYears")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer()\
                 .setInputCol ("AgeGroup")\
                 .setOutputCol ("AgeIndex")

Age_udfIndex_model=stringIndexer.fit(df)
df=Age_udfIndex_model.transform(df)

# COMMAND ----------

stringIndexer = StringIndexer()\
                 .setInputCol ("YearsAtCompanyGroup")\
                 .setOutputCol ("YearsAtCompanyIndex")

Age_udfIndex_model=stringIndexer.fit(df)
df=Age_udfIndex_model.transform(df)

# COMMAND ----------

stringIndexer = StringIndexer()\
                 .setInputCol ("TotalWorkingYearsGroup")\
                 .setOutputCol ("TotalWorkingYearsIndex")

Age_udfIndex_model=stringIndexer.fit(df)
df=Age_udfIndex_model.transform(df)

# COMMAND ----------

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing required Libaraies

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

df_0 = df

# COMMAND ----------

raw_data = df_0.toPandas ()
display(raw_data)

# COMMAND ----------

# dimentions of the raw data
print(raw_data.shape)
# first 5 rows of the data
raw_data.head(5)

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

# MAGIC %md
# MAGIC # Data Pre-Processing

# COMMAND ----------

# Limiting the data
raw_data2 = raw_data[['AgeIndex', 'Attrition', 'Gender', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYearsIndex', 'TrainingTimesLastYear', 'YearsAtCompanyIndex',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]


# Visualize the data using seaborn Pairplots
g = sns.pairplot(raw_data2, hue = 'Attrition', diag_kws={'bw': 0.2})

# COMMAND ----------

# Investigate all the features by our y

features = ['AgeIndex', 'Gender', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYearsIndex', 'TrainingTimesLastYear', 'YearsAtCompanyIndex',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']


for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=raw_data2, hue = 'Attrition', palette="Set1")

# COMMAND ----------

raw_data2.head()

# COMMAND ----------

raw_data2.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing

# COMMAND ----------

# Making categorical variables into numeric representation

new_raw_data = pd.get_dummies(raw_data2, columns = ['MaritalStatus', 'Gender'])
new_raw_data.head()

# COMMAND ----------

new_raw_data = new_raw_data.replace(to_replace = ['Yes','No'],value = ['1','0']) 

# COMMAND ----------

new_raw_data.head()

# COMMAND ----------

# Scaling our columns

scale_vars = ['MonthlyIncome']
scaler = MinMaxScaler()
new_raw_data[scale_vars] = scaler.fit_transform(new_raw_data[scale_vars])
new_raw_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Splitting Train and Test

# COMMAND ----------

X = new_raw_data.drop('Attrition', axis=1).values# Input features (attributes)
y = new_raw_data['Attrition'].values # Target vector
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size=0.3, random_state=0)

# COMMAND ----------

dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
dt.fit(X_train, y_train)

# COMMAND ----------

tree.plot_tree(dt)

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
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']               
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# COMMAND ----------

y_pred = dt.predict(X_train)
y_pred
confusion_matrix(y_train, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC # saving the model for further deployement

# COMMAND ----------

import pickle

# save
with open('model.pkl','wb') as f:
    pickle.dump(dt,f)
