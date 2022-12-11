import numpy as np
import pandas as pd
import cv2
import logging
import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()
import numpy as np
import cv2
import glob
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from functools import reduce
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.config('spark.jars.packages', 'databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11').getOrCreate()

import sparkdl
from sparkdl import DeepImageFeaturizer


# read all the images in effected folder (this has all the eye images that are effected with diabetic retinopathy)
filepath='/data/effected/*'
imagez1=[]
for filepath in glob.iglob(filepath):
    imagez1.append(ImageSchema.readImages(filepath).withColumn('label', lit(1)))
df1 = reduce(lambda first, second: first.union(second), imagez1)
df1 = df1.repartition(200)

# read all the images in effected folder (this has all the eye images that are not effected with diabetic retinopathy)
imagez2=[]
filepath='/data/uneffected/*'
for filepath in glob.iglob(filepath):
    imagez2.append(ImageSchema.readImages(filepath).withColumn('label', lit(0)))
df2 = reduce(lambda first, second: first.union(second), imagez2)
df2 = df2.repartition(200)

#combining df1 which has the data for effected eye images and df2 which has the data for un effected eye images
df2=df2.union(df1)


# model: InceptionV3
# extracting feature from images
featurizer = DeepImageFeaturizer(inputCol="image",
                                 outputCol="features",
                                 modelName="InceptionV3")

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df2.randomSplit([0.7, 0.3])

#logistic regression supports binomial classification, hence used in this project
lr = LogisticRegression(maxIter=5, regParam=0.03,elasticNetParam=0.5, labelCol="label")

# defining the pipeline model
sparkdn = Pipeline(stages=[featurizer, lr])
spark_model = sparkdn.fit(trainingData) # start fitting or training

# test the model with the testdata that is divided to get the accuracy of the model
evaluator = MulticlassClassificationEvaluator() 
tx_test = spark_model.transform(testData)
print('Accuracy ', evaluator.evaluate(tx_test, {evaluator.metricName: 'accuracy'}))
print('average precision score', evaluator.evaluate(tx_test,{evaluator.metricName:'weightedPrecision'}))
