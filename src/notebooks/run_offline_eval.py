# Databricks notebook source
# MAGIC %md
# MAGIC # Offline Eval Notebook

# COMMAND ----------
# MAGIC %run ./install_libraries $task_dependencies="SQLAlchemy==2.0.25"

# COMMAND ----------
from databricks.rag import DefaultExtendedConfig
from databricks.rag.utils import delimit_qualified_name
from databricks.rag.evaluation import run_eval, generate_offline_predictions
import mlflow
import os

# COMMAND ----------
dbutils.widgets.removeAll()

dbutils.widgets.text("config_root", "", "Configuration root path")
dbutils.widgets.text("config_file", "", "Configuration file path")
dbutils.widgets.text("environment", "", "The current environment")
dbutils.widgets.text("eval_dataset", "", "The dataset to run the evaluation against")
dbutils.widgets.text("version", "", "The version of the chain model to run evaluation on")

config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")
eval_dataset_name = dbutils.widgets.get("eval_dataset")
version = dbutils.widgets.get("version")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inputs

# COMMAND ----------

import dataclasses

extended_config = DefaultExtendedConfig(
    config_root, config_file, environment
)
input_config = extended_config.input_config
global_config = input_config.global_config
environment_info = extended_config.deployed_environment_info
experiment_name = global_config.mlflow_experiment_name
metric_conf = extended_config.metric_confs
model_name = environment_info.model_name

experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

security_scope = environment_info.security_scope
security_key = environment_info.security_key
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(security_scope, security_key)

# COMMAND ----------
print("Running run_eval")
print('eval_dataset_name', eval_dataset_name)
print('model_name', model_name)
print('version', version)
print('metric_conf', metric_conf)
print('experiment_id', experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the langchain model

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
chain = mlflow.langchain.load_model(f"models:/{model_name}/{version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the eval dataset

# COMMAND ----------

eval_dataset = spark.read.format("delta").table(delimit_qualified_name(eval_dataset_name))

# COMMAND ----------

from databricks.rag.utils import update_dataset_version_state, get_dataset_version
eval_dataset_version = get_dataset_version(delimit_qualified_name(eval_dataset_name))
write_mode = update_dataset_version_state(experiment_id, eval_dataset_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the offline predictions

# COMMAND ----------

request_log = generate_offline_predictions(
  eval_dataset, chain, model_name, version
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run eval on offline request log

# COMMAND ----------

assessment_log = run_eval(request_log, metric_conf, with_ground_truth=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write outputs

# COMMAND ----------

request_log_path = delimit_qualified_name(f"{eval_dataset_name}_request_log")
assessment_log_path = delimit_qualified_name(f"{eval_dataset_name}_assessment_log")

# COMMAND ----------

request_log.write.format("delta").mode(write_mode).saveAsTable(request_log_path)
assessment_log.write.format("delta").mode(write_mode).saveAsTable(assessment_log_path)

