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

# print Schema dataset
print('\nprintSchema:\n')
df.printSchema()

# statistical Data Analysis
print('statistical Data Analysis:\n')
df.describe().show()


# print table show column have null value
print('\n\\table show column have null value:\n')
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

# Duplicate 
print('\n\nTotal Rows is: ' , df.count())
print('Total Duplicates-Rows is: ' , df.count() - df.dropDuplicates().count())
df = df.dropDuplicates()
print('Total Rows After Delete Duplicates-Rows is: ' , df.count())

