import numpy as np
import pandas as pd
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as Func

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
spark = SparkSession.builder.getOrCreate()

training_set = spark.read.csv("../data/train.csv", sep = "\t", header = True)
testing_set = spark.read.csv("../data/test.csv", sep = "\t", header = True)

@Func.udf(returnType = StringType())
def getUnitName(problemHierarchy): 
    return problemHierarchy.split(",")[0]
@Func.udf(returnType = StringType())
def getSectionName(problemHierarchy):
    return problemHierarchy.split(",")[1]

def operateOnBoth(func, col1, col2):
    global training_set, testing_set
    training_set = training_set.withColumn(col1, func(col2))
    testing_set = testing_set.withColumn(col1, func(col2))
    
operateOnBoth(getUnitName, "Problem Unit", "Problem Hierarchy")
operateOnBoth(getSectionName, "Problem Section", "Problem Hierarchy")

def dropOnBoth(col):
    global training_set, testing_set
    training_set = training_set.drop(col)
    testing_set = testing_set.drop(col)

useless_cols = ["Row", "Problem Hierarchy", "Step Start Time", "First Transaction Time", 
                "Correct Transaction Time", "Step End Time", "Step Duration (sec)", 
                "Correct Step Duration (sec)", "Error Step Duration (sec)", 
                "Incorrects", "Hints", "Corrects"]
for col in useless_cols:
    dropOnBoth(col)

def renameOnBoth(col, new):
    global training_set, testing_set
    training_set = training_set.withColumnRenamed(col, new)
    testing_set = testing_set.withColumnRenamed(col, new)

def convertOnBoth(col):
    global training_set, testing_set
    tmp = {}
    for i, origin in enumerate(training_set.union(testing_set).select(col).distinct().collect()):
        tmp[origin[col]] = i
    
    @Func.udf(returnType = IntegerType())
    def convert(c):
        return tmp[c]
    
    operateOnBoth(convert, col+"tmp", col)
    dropOnBoth(col)
    renameOnBoth(col+"tmp", col)

convert_cols = ["Anon Student Id", "Problem Name", 
                "Problem Unit", "Problem Section", "Step Name"]
for c in convert_cols:
    convertOnBoth(c)

@Func.udf(returnType = IntegerType())
def getKCCount(kcs):
    if isinstance(kcs, str) and kcs:
        return len(kcs.split("~~"))
    else:
        return 0
@Func.udf(returnType = FloatType())
def getOpportunityAverage(oppo):
    if isinstance(oppo, str) and oppo:
        opportunities = oppo.split("~~")
    else:
        opportunities = []
        return 0.0
    opportunityCount = 0
    for val in opportunities:
        opportunityCount += int(val)
    return (1.0 * opportunityCount) / len(opportunities)

operateOnBoth(getKCCount, "KC Count", "KC(Default)")
operateOnBoth(getOpportunityAverage, "Opportunity Average", "Opportunity(Default)")
dropOnBoth("Opportunity(Default)")


good_set = training_set.filter(training_set["Correct First Attempt"] == "1")

def getCFARTemplate(CFAR, old_col):
    global training_set, testing_set
    good_collect = good_set.groupBy(old_col).count().collect()
    all_collect = training_set.groupBy(old_col).count().collect()
    good_dict = dict(good_collect)
    all_dict = dict(all_collect)
    tmp = {}
    for info in good_collect:
        tmp[info[old_col]] = (1.0 * info["count"]) / all_dict[info[old_col]]
    for info in all_collect:
        if info[old_col] not in tmp:
            tmp[info[old_col]] = 0

    @Func.udf(returnType = FloatType())
    def getR(k):
        if k not in tmp.keys():
            return float(sum(tmp.values())) / len(tmp)
        else:
            return float(tmp[k])
    
    operateOnBoth(getR, CFAR, old_col)

getCFARTemplate("Student CFAR", "Anon Student Id")
getCFARTemplate("Problem CFAR", "Problem Name")
getCFARTemplate("Unit CFAR", "Problem Unit")
getCFARTemplate("Section CFAR", "Problem Section")
getCFARTemplate("Step CFAR", "Step Name")
getCFARTemplate("KC CFAR", "KC(Default)")
dropOnBoth("KC(Default)")

training_set.toPandas().to_csv("./train_tmp.csv", sep = "\t", header = True, index = False)
testing_set.toPandas().to_csv("./test_tmp.csv", sep = '\t', header = True, index = False)

# def getPCFAR():
#     global training_set,testing_set
#     good_collect = good_set.groupBy("Anon Student Id").count().collect()
#     all_collect = training_set.groupBy("Anon Student Id").count().collect()
#     good_dict = dict(good_collect)
#     all_dict = dict(all_collect)
#     tmp = {}
#     for info in good_collect:
#         tmp[info["Anon Student Id"]] = (1.0 * info["count"]) / all_dict[info["Anon Student Id"]]
#     for info in all_collect:
#         if info["Anon Student Id"] not in tmp:
#             tmp[info["Anon Student Id"]] = 0

#     @Func.udf(returnType = FloatType())
#     def getR(k):
#         if k not in tmp.keys():
#             return float(sum(tmp.values())) / len(tmp)
#         else:
#             return float(tmp[k])
            
#     operateOnBoth(getR, "Personal CFAR", "Anon Student Id")

# getPCFAR()

# def getProCFAR():
#     global training_set,testing_set
#     good_collect = good_set.groupBy("Problem Name").count().collect()
#     all_collect = training_set.groupBy("Problem Name").count().collect()
#     good_dict = dict(good_collect)
#     all_dict = dict(all_collect)
#     tmp = {}
#     for info in good_collect:
#         tmp[info["Problem Name"]] = (1.0 * info["count"]) / all_dict[info["Problem Name"]]
#     for info in all_collect:
#         if info["Problem Name"] not in tmp:
#             tmp[info["Problem Name"]] = 0

#     @Func.udf(returnType = FloatType())
#     def getR(k):
#         if k not in tmp.keys():
#             return float(sum(tmp.values())) / len(tmp)
#         else:
#             return float(tmp[k])
    
#     operateOnBoth(getR, "Problem CFAR", "Problem Name")

# getProCFAR()
