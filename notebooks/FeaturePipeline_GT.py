# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/Amplify/MLCoreSDK-0.5.96-py3-none-any.whl --force-reinstall

# COMMAND ----------

# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install sparkmeasure

# COMMAND ----------

# DBTITLE 1,Uncomment in case you want to register the data in feature store.
from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

# dbutils.library.restartPython() 

# COMMAND ----------

try : 
    env = dbutils.widgets.get("env")
    task = dbutils.widgets.get("task")
except :
    env, task = "dev","fe"
print(f"Input environment : {env}")
print(f"Input task : {task}")

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
from MLCORE_SDK import mlclient

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

if task.lower() == "fe":
    # JOB SPECIFIC PARAMETERS FOR FEATURE PIPELINE
    ground_truth_dbfs_path = solution_config["feature_pipelines"]["feature_pipelines_gt"]["ground_truth_dbfs_path"]
    transformed_ground_truth_table_name = solution_config["feature_pipelines"]["feature_pipelines_gt"]["transformed_ground_truth_table_name"]
    is_scheduled = solution_config["feature_pipelines"]["feature_pipelines_gt"]["is_scheduled"]
    batch_size = int(solution_config["feature_pipelines"]["feature_pipelines_gt"].get("batch_size",500))
    cron_job_schedule = solution_config["feature_pipelines"]["feature_pipelines_gt"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["feature_pipelines"]["feature_pipelines_gt"]["primary_keys"]
else: 
    # JOB SPECIFIC PARAMETERS FOR DATA PREP DEPLOYMENT
    ground_truth_dbfs_path = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["ground_truth_dbfs_path"]
    transformed_ground_truth_table_name = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["transformed_ground_truth_table_name"]
    is_scheduled = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["is_scheduled"]
    batch_size = int(solution_config["data_prep_deployments"]["data_prep_deployment_gt"].get("batch_size",500))
    cron_job_schedule = solution_config["data_prep_deployments"]["data_prep_deployment_gt"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["primary_keys"]

    mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = task)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### FEATURE ENGINEERING

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### FEATURE ENGINEERING on Ground Truth Data

# COMMAND ----------

# DBTITLE 1,Load the Data
ground_truth_df = spark.sql(f"SELECT * FROM {db_name}.{ground_truth_dbfs_path}")

# COMMAND ----------

from pyspark.sql import functions as F
import pickle

# COMMAND ----------

if is_scheduled:
  # CREATE PICKLE FILE
  pickle_file_path = f"/mnt/FileStore/{db_name}"
  dbutils.fs.mkdirs(pickle_file_path)
  print(f"Created directory : {pickle_file_path}")
  pickle_file_path = f"/dbfs/{pickle_file_path}/{transformed_ground_truth_table_name}.pickle"

  # LOAD CACHE IF AVAILABLE
  try : 
    with open(pickle_file_path, "rb") as handle:
        obj_properties = pickle.load(handle)
        print(f"Instance loaded successfully")
  except Exception as e:
    print(f"Exception while loading cache : {e}")
    obj_properties = {}
  print(f"Existing Cache : {obj_properties}")

  if not obj_properties :
    start_marker = 1
  elif obj_properties and obj_properties.get("end_marker",0) == 0:
    start_marker = 1
  else :
    start_marker = obj_properties["end_marker"] + 1
  end_marker = start_marker + batch_size - 1

  print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")
else :
  start_marker = 1
  end_marker = ground_truth_df.count()


# COMMAND ----------

# DBTITLE 1,Perform some feature engineering step.
GT_DF = ground_truth_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

# DBTITLE 1,Exit the job if there is no new data
if not GT_DF.first():
  dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

data = GT_DF.toPandas()

# COMMAND ----------

data.display()

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
# Define the columns to exclude from min-max scaling
exclude_columns = ['Index', 'timestamp', 'date', 'id']

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Apply min-max scaling to the DataFrame while excluding specified columns
scaled_columns = data.drop(columns=exclude_columns).columns
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

# COMMAND ----------

GT_DF = spark.createDataFrame(data)

# COMMAND ----------

GT_DF.display()

# COMMAND ----------

GT_DF.createOrReplaceTempView(transformed_ground_truth_table_name)

gt_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == transformed_ground_truth_table_name.lower() and not table_data.isTemporary]

if not any(gt_table_exist):
    print(f"CREATING TABLE")
    spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{transformed_ground_truth_table_name} AS SELECT * FROM {transformed_ground_truth_table_name}")
else :
    print(F"UPDATING TABLE")
    spark.sql(f"INSERT INTO hive_metastore.{db_name}.{transformed_ground_truth_table_name} SELECT * FROM {transformed_ground_truth_table_name}")

# COMMAND ----------

from pyspark.sql import functions as F
gt_hive_table_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{transformed_ground_truth_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(f"Ground Truth Hive Path : {gt_hive_table_path}")

# COMMAND ----------

stagemetrics.end()
stagemetrics.print_report()

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### REGISTER THE FEATURES ON MLCORE
# MAGIC

# COMMAND ----------

# DBTITLE 1,Register Ground Truth Transformed Table
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = transformed_ground_truth_table_name,
    num_rows = GT_DF.count(),
    cols = GT_DF.columns,
    column_datatype = GT_DF.dtypes,
    table_schema = GT_DF.schema,
    primary_keys = primary_keys,
    table_path = gt_hive_table_path,
    table_sub_type="Ground_Truth",
    request_type = task,
    env = env,
    batch_size = batch_size,
    quartz_cron_expression = cron_job_schedule,
    compute_usage_metrics = compute_metrics,
    verbose = True)

# COMMAND ----------

if is_scheduled:
  obj_properties['end_marker'] = end_marker
  with open(pickle_file_path, "wb") as handle:
      pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Instance successfully saved successfully")
