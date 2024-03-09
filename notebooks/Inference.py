# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/Amplify/MLCoreSDK-0.5.96-py3-none-any.whl --force-reinstall

# COMMAND ----------

# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Imports
from MLCORE_SDK import mlclient
import ast
from pyspark.sql import functions as F
from datetime import datetime
from delta.tables import *
import time
import pandas as pd
import mlflow
import pickle

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Model & Table parameters
# MAGIC

# COMMAND ----------

import yaml
with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']
use_latest = True

# JOB SPECIFIC PARAMETERS FOR INFERENCE
transformed_features_path = solution_config["inference"]["transformed_features_path"]
ground_truth_path = solution_config["inference"]["ground_truth_path"]
output_table_name = solution_config["inference"]["output_table_name"]
batch_size = int(solution_config["inference"].get("batch_size",500))
cron_job_schedule = solution_config["inference"].get("cron_job_schedule","0 */10 * ? * *")
model_name = solution_config["inference"]["model_name"]
model_version = solution_config["inference"]["model_version"]
feature_columns = solution_config['train']["feature_columns"]
date_column = solution_config['train']["date_column"]
primary_keys = solution_config["train"]["primary_keys"]
target_columns = solution_config['train']["target_columns"]

# COMMAND ----------

if use_latest:
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    x = client.get_latest_versions(model_name)
    model_version = x[0].version
    print(model_version)

# COMMAND ----------

pickle_file_path = f"/mnt/FileStore/{db_name}"
dbutils.fs.mkdirs(pickle_file_path)
print(f"Created directory : {pickle_file_path}")
pickle_file_path = f"/dbfs/{pickle_file_path}/{output_table_name}.pickle"

# COMMAND ----------

try : 
  with open(pickle_file_path, "rb") as handle:
      obj_properties = pickle.load(handle)
      print(f"Instance loaded successfully")
except Exception as e:
  print(f"Exception while loading cache : {e}")
  obj_properties = {}
print(f"Existing Cache : {obj_properties}")

# COMMAND ----------

if not obj_properties :
  start_marker = 1
elif obj_properties and obj_properties.get("end_marker",0) == 0:
  start_marker = 1
else :
  start_marker = obj_properties["end_marker"] + 1
end_marker = start_marker + batch_size - 1

print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

features_df = spark.sql(f"SELECT * FROM {db_name}.{transformed_features_path}")
gt_data = spark.sql(f"SELECT * FROM {db_name}.{ground_truth_path}")

# COMMAND ----------

features_df.display()

# COMMAND ----------

FT_DF = features_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))
GT_DF = gt_data.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

if not FT_DF.first():
  dbutils.notebook.exit("No data is available for inference, hence exiting the notebook")

# COMMAND ----------

from MLCORE_SDK import mlclient
mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = "inference")

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Load Inference Features Data

# COMMAND ----------

ground_truth = gt_data.toPandas()[primary_keys + target_columns]
tranformed_features_df = FT_DF.toPandas()
inference_df = tranformed_features_df[[date_column] + feature_columns]
inference_df = inference_df.rename(columns={date_column: "ds"})
display(inference_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Model

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

predictions = loaded_model.predict(inference_df)
type(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Output Table

# COMMAND ----------

tranformed_features_df["prediction"] = predictions["yhat"]
tranformed_features_df = pd.merge(tranformed_features_df,ground_truth, on=primary_keys[0], how='left')
output_table = spark.createDataFrame(tranformed_features_df)

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_job_helper import get_job_id_run_id

job_id, run_id = get_job_id_run_id(dbutils)

output_table = output_table.withColumnRenamed(target_columns[0],"ground_truth_value")
output_table = output_table.withColumn("acceptance_status",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_time",F.lit(None).cast("long"))
output_table = output_table.withColumn("accepted_by_id",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_by_name",F.lit(None).cast("string"))
output_table = output_table.withColumn("moderated_value",F.lit(None).cast("double"))
output_table = output_table.withColumn("inference_job_id",F.lit(job_id).cast("string"))
output_table = output_table.withColumn("inference_run_id",F.lit(run_id).cast("string"))

# COMMAND ----------

output_table.display()

# COMMAND ----------

output_table = output_table.drop('date','timestamp')

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_table = output_table.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_table = output_table.withColumn("date", F.lit(date))
output_table = output_table.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in output_table.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  output_table = output_table.withColumn("id", F.row_number().over(window))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Save Output Table

# COMMAND ----------

output_table.createOrReplaceTempView(output_table_name)

output_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == output_table_name.lower() and not table_data.isTemporary]

if not any(output_table_exist):
  print(f"CREATING TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{output_table_name} AS SELECT * FROM {output_table_name}")
else :
  print(F"UPDATING TABLE")
  spark.sql(f"INSERT INTO hive_metastore.{db_name}.{output_table_name} SELECT * FROM {output_table_name}");

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Output Table dbfs path

# COMMAND ----------

output_hive_table_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{output_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(f"output Hive Path : {output_hive_table_path}")

# COMMAND ----------

feature_table_dbfs_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{transformed_features_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
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
# MAGIC ###Register Inference artifacts

# COMMAND ----------

mlclient.log(operation_type = "register_inference",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    output_table_name=output_table_name,
    output_table_path=output_hive_table_path,
    feature_table_path=feature_table_dbfs_path,
    ground_truth_table_path=gt_table_dbfs_path,
    model_name=model_name,
    model_version=model_version,
    num_rows=output_table.count(),
    cols=output_table.columns,
    table_type="internal",
    batch_size = str(batch_size),
    env = env,
    column_datatype = output_table.dtypes,
    table_schema = output_table.schema,
    compute_usage_metrics = compute_metrics,
    verbose = True)

# COMMAND ----------

obj_properties['end_marker'] = end_marker
with open(pickle_file_path, "wb") as handle:
    pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Instance successfully saved successfully")
