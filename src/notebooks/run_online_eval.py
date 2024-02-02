# Databricks notebook source
# MAGIC %run ./install_libraries

# COMMAND ----------
# DBTITLE 1,Set widgets for required parameters of this notebook
dbutils.widgets.text("config_root", "", label="Configuration root path")
dbutils.widgets.text("config_file", "", label="Configuration file path")
dbutils.widgets.text("environment", "", label="The current environment")

config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")

# COMMAND ----------

# DBTITLE 1,Derive the configurations to run this notebook
from databricks.rag import DefaultExtendedConfig
from databricks.rag.utils import delimit_qualified_name, get_table_url
from databricks.rag.evaluation import (
    dedup_assessment_logs, get_payload_table_name,
    persist_stream, run_eval, unpack_and_split_payloads
)


config = DefaultExtendedConfig(
    root_path=config_root,
    config_path=config_file,
    env_name=environment,
)

target_catalog = config.input_config.global_config.uc_assets_location.catalog
target_schema = config.input_config.global_config.uc_assets_location.schema
metric_confs = config.metric_confs
endpoint_name = config.deployed_environment_info.endpoint_name
payload_table_name = get_payload_table_name(endpoint_name)
request_log_name = delimit_qualified_name(config.deployed_environment_info.request_log_table)
assessment_log_name = delimit_qualified_name(config.deployed_environment_info.assessment_log_table)

print(f"""Payload table name: {payload_table_name}
Request log name: {request_log_name}
Assessment log name: {assessment_log_name}""")

# COMMAND ----------

# DBTITLE 1,Initialize the streaming checkpoints
# Create a Volume to store streaming checkpoints
volume_name = f"{environment}_checkpoints"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {target_catalog}.{target_schema}.{volume_name}")

request_log_checkpoint_location = f"/Volumes/{target_catalog}/{target_schema}/{volume_name}/{request_log_name}/checkpoints"
assessment_log_checkpoint_location = f"/Volumes/{target_catalog}/{target_schema}/{volume_name}/{assessment_log_name}/checkpoints"

# COMMAND ----------

# DBTITLE 1,Unpack the payloads as a stream
# Read in the payloads as a stream and unpack them.
payload_df = spark.readStream.table(payload_table_name)
request_logs, assessment_logs = unpack_and_split_payloads(payload_df)

# COMMAND ----------

# DBTITLE 1,Run online evaluation
# Calculate evaluation assessments on the request logs, and concatenate them with logged assessments.
eval_assessments = run_eval(request_logs, metric_confs)
all_assessments = assessment_logs.unionByName(eval_assessments)

# COMMAND ----------

# DBTITLE 1,Persist the results
# Persist the streams, with defined checkpoint paths for these table.
persist_stream(request_log_name, request_logs, request_log_checkpoint_location)
persist_stream(assessment_log_name, all_assessments, assessment_log_checkpoint_location)

# COMMAND ----------

# DBTITLE 1,Deduplicate assessments
# Finally, deduplicate assessment logs globally to handle updates within the same hour.
assessments_df = spark.table(assessment_log_name)
deduped_assessments_df = dedup_assessment_logs(assessments_df, granularity="hour")
deduped_assessments_df.write.format("delta").mode("overwrite").saveAsTable(assessment_log_name)

# COMMAND ----------

# DBTITLE 1,Exit and print the table names
workspace_url = config.input_config.global_config.workspace_url
request_log_url = get_table_url(workspace_url, config.deployed_environment_info.request_log_table)
assessment_log_url = get_table_url(workspace_url, config.deployed_environment_info.assessment_log_table)
dbutils.notebook.exit(
    f"View the request log at: {request_log_url}\n"
    f"View the assessment log at: {assessment_log_url}"
)
