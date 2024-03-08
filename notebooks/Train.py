# Databricks notebook source
# MAGIC %md
# MAGIC ## INSTALL MLCORE SDK

# COMMAND ----------

# MAGIC %pip install /dbfs/FileStore/Amplify/MLCoreSDK-0.5.96-py3-none-any.whl --force-reinstall

# COMMAND ----------

# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PERFORM MODEL TRAINING 

# COMMAND ----------

# DBTITLE 1,Imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import *
import json
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from prophet import Prophet
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# DBTITLE 1,Input from the user
# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# JOB SPECIFIC PARAMETERS
feature_table_path = solution_config['train']["feature_table_path"]
ground_truth_path = solution_config['train']["ground_truth_path"]
primary_keys = solution_config['train']["primary_keys"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
date_column = solution_config['train']["date_column"]
horizon = solution_config['train']["horizon"]
frequency = solution_config['train']["frequency"]
train_output_table_name = solution_config['train']["train_output_table_name"]
test_size = solution_config['train']["test_size"]
model_name = solution_config['train']["model_name"]
model_version = solution_config['train']["model_version"]

# COMMAND ----------

# DBTITLE 1,Update the table paths as needed.
ft_data = spark.sql(f"SELECT * FROM {db_name}.{feature_table_path}")
gt_data = spark.sql(f"SELECT * FROM {db_name}.{ground_truth_path}")

# COMMAND ----------

try : 
    date_filters = dbutils.widgets.get("date_filters")
    print(f"Input date filter : {date_filters}")
    date_filters = json.loads(date_filters)
except :
    date_filters = {}

try : 
    hyperparameters = dbutils.widgets.get("hyperparameters")
    print(f"Input hyper parameters : {hyperparameters}")
    hyperparameters = json.loads(hyperparameters)
except :
    hyperparameters = {}

print(f"Data filters used in model train : {date_filters}, hyper parameters : {hyperparameters}")


# COMMAND ----------

if date_filters and date_filters['feature_table_date_filters'] and date_filters['feature_table_date_filters'] != {} : 
    print(f"Filtering the feature data")
    if date_filters['feature_table_date_filters']['start_date'] and date_filters['feature_table_date_filters']['start_date'] != "" : 
        ft_data = ft_data.filter(F.col("date") >= date_filters['feature_table_date_filters']['start_date']).filter(F.col("date") <= date_filters['feature_table_date_filters']['end_date'])

if date_filters and date_filters['ground_truth_table_date_filters'] and date_filters['ground_truth_table_date_filters'] != {} : 
    print(f"Filtering the ground truth data")
    if date_filters['ground_truth_table_date_filters']['start_date'] and date_filters['ground_truth_table_date_filters']['start_date'] != "": 
        gt_data = gt_data.filter(F.col("date") >= date_filters['ground_truth_table_date_filters']['start_date']).filter(F.col("date") <= date_filters['ground_truth_table_date_filters']['end_date'])

# COMMAND ----------

features_data = ft_data.select(primary_keys + [date_column] + feature_columns)
ground_truth_data = gt_data.select(primary_keys + target_columns)

# COMMAND ----------

# DBTITLE 1,Joining Feature and Ground truth tables on primary key
final_df = features_data.join(ground_truth_data, on = primary_keys)

# COMMAND ----------

# DBTITLE 1,Converting the Spark df to Pandas df
final_df_pandas = final_df.toPandas()
# final_df_pandas.head()

# COMMAND ----------

# DBTITLE 1,Dropping the null rows in the final df
final_df_pandas.dropna(inplace=True)

# COMMAND ----------

final_df_pandas.display()

# COMMAND ----------

# DBTITLE 1,Spliting the Final df to test and train dfs
# Split the Data to Train and Test
traindf = final_df_pandas.iloc[int(final_df_pandas.shape[0] * test_size):]
testdf = final_df_pandas.iloc[:int(final_df_pandas.shape[0] * test_size)]

# COMMAND ----------

# try :
#     hp_tuning_result = dbutils.notebook.run("Hyperparameter_Tuning", timeout_seconds = 0)
#     hyperparameters = json.loads(hp_tuning_result)["best_hyperparameters"]
# except :
#     hp_tuning_result

# COMMAND ----------

# DBTITLE 1,Defining the Model

if not hyperparameters or hyperparameters == {} :
    model = Prophet()
    print(f"Using model with default hyper parameters")
else :
    model = Prophet(**hyperparameters)
    print(f"Using model with custom hyper parameters")

# COMMAND ----------

for feature in feature_columns:
    model.add_regressor(feature, standardize=False)

# COMMAND ----------

train_prophet_df = traindf.rename(columns={date_column: "ds", target_columns[0]: "y"})
test_prophet_df = testdf.rename(columns={date_column: "ds", target_columns[0]: "y"})

# COMMAND ----------

# DBTITLE 1,Fitting the model on Train data
model = model.fit(train_prophet_df, iter=200)

# COMMAND ----------

# DBTITLE 1,Fetching Train and test predictions from model
y_train = traindf[target_columns[0]]
y_test = testdf[target_columns[0]]

# Predict
train_pred = model.predict(train_prophet_df)
test_pred = model.predict(test_prophet_df)

# COMMAND ----------

# DBTITLE 1,Displaying Forecast plots
model.plot_components(train_pred)

# COMMAND ----------

# DBTITLE 1,Get prediction columns
y_pred_train = train_pred["yhat"].to_numpy()
y_pred = test_pred["yhat"].to_numpy()

# COMMAND ----------

# DBTITLE 1,Calculating the test metrics from the model
# Predict it on Test and calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# COMMAND ----------

# DBTITLE 1,Displaying the test metrics 
test_metrics = {"r2":r2, "mse":mse, "mae":mae, "rmse":rmse}
test_metrics

# COMMAND ----------

# DBTITLE 1,Calculating the train metrics from the model
# Predict it on Test and calculate metrics
r2 = r2_score(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train, squared=False)

# COMMAND ----------

# DBTITLE 1,Displaying the train metrics 
train_metrics = {"r2":r2, "mse":mse, "mae":mae, "rmse":rmse}
train_metrics

# COMMAND ----------

# DBTITLE 1,Saving the predictions in "yhat" column
pred_train = traindf
pred_train["prediction"] = y_pred_train

pred_test = testdf
pred_test["prediction"] = y_pred

# COMMAND ----------

# DBTITLE 1,Saving the Actual target values in "y" column
pred_test[target_columns[0]] = y_test
pred_train[target_columns[0]] = y_train

# COMMAND ----------

columns_for_reports = ["trend", "daily", "yearly", "weekly"]

# COMMAND ----------

if "daily" not in train_pred.columns:
    columns_for_reports.remove("daily")

if "weekly" not in train_pred.columns:
    columns_for_reports.remove("weekly")

if "yearly" not in train_pred.columns:
    columns_for_reports.remove("yearly")

print(f"Columns considered for report : {columns_for_reports}")

# COMMAND ----------

for report_column in columns_for_reports:
    if report_column in train_pred.columns:
        pred_train[report_column] = train_pred[report_column].to_numpy()
        pred_test[report_column] = test_pred[report_column].to_numpy()

# COMMAND ----------

# pred_train

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE PREDICTIONS TO HIVE

# COMMAND ----------

pred_train["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "train"
pred_test["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "test"

# COMMAND ----------

final_train_output_df = pd.concat([pred_train, pred_test])

# COMMAND ----------

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats = (
        "MM-dd-yyyy",
        "dd-MM-yyyy",
        "MM/dd/yyyy",
        "yyyy-MM-dd",
        "M/d/yyyy",
        "M/dd/yyyy",
        "MM/dd/yy",
        "MM.dd.yyyy",
        "dd.MM.yyyy",
        "yyyy-MM-dd",
        "yyyy-dd-MM",
    )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

train_output_df = spark.createDataFrame(final_train_output_df)
now = datetime.now()
date = now.strftime("%m-%d-%Y")
train_output_df = train_output_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
train_output_df = train_output_df.withColumn("date", F.lit(date))
train_output_df = train_output_df.withColumn("date", to_date_(F.col("date")))
w = Window.orderBy(F.monotonically_increasing_id())

train_output_df = train_output_df.withColumn("id", F.row_number().over(w))

# COMMAND ----------

train_output_df.display()

# COMMAND ----------

train_output_df.createOrReplaceTempView(train_output_table_name)
print(f"CREATING TABLE")
spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{train_output_table_name} AS SELECT * FROM {train_output_table_name}")

# COMMAND ----------

train_output_dbfs_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{train_output_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(train_output_dbfs_path)

# COMMAND ----------

feature_table_dbfs_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{feature_table_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
gt_table_dbfs_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{ground_truth_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(feature_table_dbfs_path, gt_table_dbfs_path)

# COMMAND ----------

stagemetrics.end()
stagemetrics.print_report()

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## REGISTER MODEL IN MLCORE

# COMMAND ----------

from MLCORE_SDK import mlclient

# COMMAND ----------

train_data_date_dict = {
    "feature_table" : {
        "ft_start_date" : ft_data.select(F.min("timestamp")).collect()[0][0],
        "ft_end_date" : ft_data.select(F.max("timestamp")).collect()[0][0]
    },
    "gt_table" : {
        "gt_start_date" : gt_data.select(F.min("timestamp")).collect()[0][0],
        "gt_end_date" : gt_data.select(F.max("timestamp")).collect()[0][0]        
    }
}

# COMMAND ----------

# DBTITLE 1,Registering the model in MLCore
mlclient.log(operation_type = "register_model",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    model = model,
    model_name = f"{model_name}",
    model_runtime_env = "python",
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    feature_table_path = feature_table_dbfs_path,
    ground_truth_table_path = gt_table_dbfs_path,
    train_output_path = train_output_dbfs_path,
    train_output_rows = train_output_df.count(),
    train_output_cols = train_output_df.columns,
    feature_columns = feature_columns,
    target_columns = target_columns,
    date_column=date_column,
    horizon=horizon,
    frequency=frequency,
    train_data_date_dict = train_data_date_dict,
    compute_usage_metrics = compute_metrics,
    verbose = True
    )

# COMMAND ----------

# try :
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils)
#     dbutils.notebook.run(
#         "Model_Test", 
#         timeout_seconds = 5000, 
#         arguments = 
#         {
#             "feature_columns" : ",".join(map(str,feature_columns)),
#             "target_columns" : ",".join(map(str,target_columns)), #json dumps
#             "model_data_path" : train_output_dbfs_path,
#             "model_name": model_name,
#             "media_artifacts_path" : media_artifacts_path,
#         })
# except Exception as e:
#     print(f"Exception while triggering model testing notebook : {e}")

# COMMAND ----------

# try: 
#     #define media artifacts path
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils)
    
#     print(media_artifacts_path)

#     custom_notebook_result = dbutils.notebook.run(
#         "Model_eval",
#         timeout_seconds = 0,
#         arguments = 
#         {
#             "date_column" : date_column,
#             "feature_columns" : ",".join(map(str,feature_columns)),
#             "target_columns" : ",".join(map(str,target_columns)), #json dumps
#             "model_data_path" : train_output_dbfs_path,
#             "model_name": model_name,
#             "media_artifacts_path" : media_artifacts_path,
#         })
# except Exception as e:
#     print(f"Exception while triggering model eval notebook : {e}")
