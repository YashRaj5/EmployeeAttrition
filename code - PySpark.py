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

# copying data from on premise to ADLS raw folder
dbutils.fs.cp("dbfs:/mnt/raw_data.csv", "/Workspace/Repos/yash.raj@hanu.com/EmployeeAttrition/data/csv", True)

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

# MAGIC %md
# MAGIC ## Importing required Libaraies

# COMMAND ----------

from matplotlib import cm
import matplotlib.pyplot as plt

from pyspark.sql.functions import isnan, when, rank, sum, count, col, desc
from pyspark.sql import Window
import pyspark.sql.functions as F

# COMMAND ----------

from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder , StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC * Distribution of Features

# COMMAND ----------

fig = plt.figure(figsize=(25,15)) # Plot size
st = fig.suptitle("Distribution of Features", fontsize=50, verticalalignment='center') # Plot main titile
for col,num in zip(df.toPandas().describe().columns, range(1,36)):
    ax = fig.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
    plt.style.use('dark_background')
    plt.grid(False)
    plt.xticks(rotation=45,fontsize=20)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace=0.4)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC * Target Value Distribution

# COMMAND ----------

df.groupby("Attrition").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC * Checking for null values

# COMMAND ----------

df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().head()

# COMMAND ----------

# extra
# for converting '0' & '1' values to 'No' & 'Yes'
#from pyspark.sql.functions import udf
#y_udf = udf(lambda y: "No" if y==0 else "yes", StringType())

#df = df.withColumn("Attrition", #y_udf('Attrition_in_Yes&No', #y_udf("Attrition")).drop("Attrition"))
#df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Creating Age Group

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

# COMMAND ----------

window = Window.rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)
tab = df.select(['AgeGroup','MonthlyIncome']).\
   groupBy('AgeGroup').\
       agg(F.count('MonthlyIncome').alias('UserCount'),
           F.mean('MonthlyIncome').alias('MonthlyIncome_AVG'),
           F.min('MonthlyIncome').alias('MonthlyIncome_MIN'),
           F.max('MonthlyIncome').alias('MonthlyIncome_MAX')).\
       withColumn('total',sum(col('UserCount')).over(window)).\
       withColumn('Percent',col('UserCount')*100/col('total')).\
       drop(col('total')).sort(desc("Percent"))

# COMMAND ----------

tab.display()

# COMMAND ----------

# Data to plot
labels = list(tab.select('AgeGroup').distinct().toPandas()['AgeGroup'])
sizes =  list(tab.select('Percent').distinct().toPandas()['Percent'])
colors = ['gold', 'yellowgreen', 'lightcoral','blue', 'lightskyblue','green','red']
explode = (0.1, 0.0, 0 ,0.0 )  # explode 1st slice

# Plot
plt.figure(figsize=(10,8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

# COMMAND ----------

df.display()

# COMMAND ----------

df=df.drop("Age")

# COMMAND ----------

# MAGIC %md
# MAGIC * Determining if there is a high correlation (*Pearson Correlation*)  between our data

# COMMAND ----------

numeric_features = [t[0] for t in df.dtypes if t[1] != 'string']
numeric_features_df=df.select(numeric_features)
numeric_features_df.toPandas().head()

# COMMAND ----------

#col_names = numeric_features_df.columns
#features = numeric_features_df.rdd.map(lambda #row:row[0:])
#corr_mat=Statistics.corr(features, method="pearson")
#corr_df = pd.DataFrame(corr_mat)
#corr_df.index, corr_df.columns = col_names, col_names
#corr_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing for modelling

# COMMAND ----------

# MAGIC %md
# MAGIC * String Indexer - converting categorical strings into numerical

# COMMAND ----------

df2=df
df3=df

# COMMAND ----------

stringIndexer = StringIndexer()\
                 .setInputCol ("Age_udf")\
                 .setOutputCol ("Age_udfIndex")

Age_udfIndex_model=stringIndexer.fit(df2)
Age_udfIndex_df=Age_udfIndex_model.transform(df2)
Age_udfIndex_df.toPandas().head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC * One Hot Encoding - converting categorical variables into binary SparseVectors

# COMMAND ----------

encoder = OneHotEncoder()\
.setInputCols(['AgeIndex'])\
.setOutputCols(['AgeEncode'])
encoder_model=encoder.fit(Age_udfIndex_df)
encoder_df=encoder_model.transform(Age_udfIndex_df)

encoder_df.toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Vector Assembly - for converting all the features in vertors

# COMMAND ----------

import pandas as pd
pd.set_option('display.max_colwidth', 80)
pd.set_option('max_columns', 12)

# COMMAND ----------

assembler = VectorAssembler()\
         .setInputCols (["AgeEncode","DistanceFromHome","Education",
                         "EnvironmentSatisfaction","EnvironmentSatisfaction",                         "JobInvolvement","JobSatisfaction","MonthlyIncome","NumCompaniesWorked","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"])\
         .setOutputCol ("vectorized_features")
        

assembler_df=assembler.transform(encoder_df)
assembler_df.toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Label Indexing - Convert label into label indices using the StringIndexer

# COMMAND ----------

label_indexer = StringIndexer()\
         .setInputCol ("Attrition")\
         .setOutputCol ("label")

label_indexer_model=label_indexer.fit(assembler_df)
label_indexer_df=label_indexer_model.transform(assembler_df)

label_indexer_df.select("Attrition","label").toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Standardizing the dataset

# COMMAND ----------

scaler = StandardScaler()\
         .setInputCol ("vectorized_features")\
         .setOutputCol ("features")
        
scaler_model=scaler.fit(label_indexer_df)
scaler_df=scaler_model.transform(label_indexer_df)
pd.set_option('display.max_colwidth', 40)
scaler_df.select("vectorized_features","features").toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Pipeline

# COMMAND ----------

pipeline_stages=Pipeline()\
                .setStages([stringIndexer,encoder,assembler,label_indexer,scaler])
pipeline_model=pipeline_stages.fit(df3)
pipeline_df=pipeline_model.transform(df3)

# COMMAND ----------

pipeline_df.toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Train / Test Split

# COMMAND ----------

train, test = pipeline_df.randomSplit([0.8, 0.2], seed=2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

train.groupby("Attrition").count().show()

# COMMAND ----------

test.groupby("Attrition").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

dtc = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dtc = dtc.fit(train)

pred = dtc.transform(test)
pred.show(3)

# COMMAND ----------

evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")
acc = evaluator.evaluate(pred)
 
print("Prediction Accuracy: ", acc)
 
y_pred=pred.select("prediction").collect()
y_orig=pred.select("label").collect()

cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm) 

# COMMAND ----------

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

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=5)
lrModel = lr.fit(train)
predictions = lrModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------

lrModel.save("abc.model")

# COMMAND ----------

sameModel = LogisticRegressionModel.load("abc.model")

# COMMAND ----------

predictions.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Confusion Matrix

# COMMAND ----------

class_names=[1.0,0.0]
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# COMMAND ----------

y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Accuracy

# COMMAND ----------

accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ",accuracy)


# COMMAND ----------

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross Validation and Parameter Tuining

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])# regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])# Elastic Net Parameter (Ridge = 0)
             .addGrid(lr.maxIter, [1, 5, 10])#Number of iterations
             .build())

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, 
                    evaluator=evaluator, numFolds=5)

cvModel = cv.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Best Model

# COMMAND ----------

## Evaluate Best Model
predictions = cvModel.transform(test)
print('Best Model Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC # Best Model Feature Weights

# COMMAND ----------

cvModel.bestModel

# COMMAND ----------

weights = cvModel.bestModel.coefficients
weights = [(float(w),) for w in weights]
weightsDF = sqlContext.createDataFrame(weights, ["Feature Weight"])
weightsDF.toPandas().head(10)


# COMMAND ----------

# MAGIC %md
# MAGIC # Best Model Parameters

# COMMAND ----------

best_model=cvModel.bestModel

# COMMAND ----------


best_model.explainParams().split("\n")
