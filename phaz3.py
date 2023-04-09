from pyspark.sql import SparkSession
import pyspark.sql.functions as funcs
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator


spark = SparkSession.builder.master("spark://am-virtual-machine:7077").appName('Machin Learning').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("-----------------------------------------------------------------------------------------")
print("\n\n\n\n\n")

df = spark.read.csv('/home/am/bigdata/ML_hw_dataset.csv', header = True , inferSchema = 'true')

# Duplicate 
df = df.dropDuplicates()


numerical_columns = []
categorical_columns = []
for cols in df.columns:
    if df.select(cols).dtypes[0][1] != "string":
        numerical_columns.append(cols)
    else:
        categorical_columns.append(cols)


# Outlier Value
fig, axs = plt.subplots(2,5)
k = df.select(numerical_columns)
dfp = k.toPandas()
sns.boxplot(dfp[numerical_columns[0]],ax=axs[0][0]).set(xlabel='age')
sns.boxplot(dfp[numerical_columns[1]],ax=axs[0][1]).set(xlabel='duration')
sns.boxplot(dfp[numerical_columns[2]],ax=axs[0][2]).set(xlabel='campaign')
sns.boxplot(dfp[numerical_columns[3]],ax=axs[0][3]).set(xlabel='pdays')
sns.boxplot(dfp[numerical_columns[4]],ax=axs[0][4]).set(xlabel='previous')
sns.boxplot(dfp[numerical_columns[5]],ax=axs[1][0]).set(xlabel='emp_var_rate')
sns.boxplot(dfp[numerical_columns[6]],ax=axs[1][1]).set(xlabel='cons_price_idx')
sns.boxplot(dfp[numerical_columns[7]],ax=axs[1][2]).set(xlabel='cons_conf_idx')
sns.boxplot(dfp[numerical_columns[8]],ax=axs[1][3]).set(xlabel='euribor3m')
sns.boxplot(dfp[numerical_columns[9]],ax=axs[1][4]).set(xlabel='nr_employed')
plt.show()

print('Count data Before Deleting Outliers: ',df.count())
def outlier(data,column):
    Q1 = df.approxQuantile(column,[0.25],relativeError=0)
    Q3 = df.approxQuantile(column,[0.75],relativeError=0)
    IQR = Q3[0] - Q1[0]
    lower =  Q1[0] - 1.5*IQR
    upper =  Q3[0] + 1.5*IQR
    data_filtered = data.withColumn(column, when(col(column) < lower, lower).when(col(column) > upper, upper).otherwise(col(column)))
    return data_filtered

df = outlier(df,numerical_columns[0])
df = outlier(df,numerical_columns[1])
df = outlier(df,numerical_columns[2])
df = outlier(df,numerical_columns[3])
df = outlier(df,numerical_columns[4])
df = outlier(df,numerical_columns[7])

print('Count data After Deleting Outliers: ',df.count())


# Convert Column Categorical to Numerical
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid = "skip") for column in categorical_columns]
pipeline = Pipeline(stages = indexers)
data_index = pipeline.fit(df).transform(df)
for col in categorical_columns:
    data_index = data_index.drop(col)


# LR
vectorAssembler = VectorAssembler(inputCols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 
'cons_conf_idx', 'euribor3m', 'nr_employed', 'job_index', 'marital_index', 'education_index', 'default_index', 
'housing_index', 'loan_index', 'contact_index', 'month_index', 'day_of_week_index', 'poutcome_index'], outputCol='features')

output = vectorAssembler.transform(data_index)
model_df = output.select('features','y')
train_data, test_data = model_df.randomSplit([0.8, 0.2])
print('Train Count: ',train_data.count())
print('Test Count: ',test_data.count())
lr = LogisticRegression(labelCol = 'y').fit(train_data)
result = lr.evaluate(test_data).predictions
result.show()

tp = result[(result.y == 1) & (result.prediction == 1)].count()
tn = result[(result.y == 0) & (result.prediction == 0)].count()
fp = result[(result.y == 0) & (result.prediction == 1)].count()
fn = result[(result.y == 1) & (result.prediction == 0)].count()

print('Accuracy: ',float((tp+tn)/(tp+tn+fp+fn)))
print('Precision: ',float((tp)/(tp+fp)))
