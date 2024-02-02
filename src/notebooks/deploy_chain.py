# Databricks notebook source
# MAGIC %run ./install_libraries

# COMMAND ----------

dbutils.widgets.removeAll()

dbutils.widgets.text("config_root", "", "Configuration root path")
dbutils.widgets.text("config_file", "", "Configuration file path")
dbutils.widgets.text("environment", "", "The current environment")
dbutils.widgets.text("version", "", "Model version to deploy")

config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")
model_version = dbutils.widgets.get("version") or dbutils.jobs.taskValues.get(
    taskKey="build_chain_task", key="model_version", default=""
)

# COMMAND ----------

from databricks.rag import DefaultExtendedConfig

extended_config = DefaultExtendedConfig(
    config_root, config_file, environment
)
environment_info = extended_config.deployed_environment_info

model_name = environment_info.model_name
serving_endpoint_name = environment_info.endpoint_name

# COMMAND ----------

print(
    f"Depolying chain model {model_name} with {model_version} to {serving_endpoint_name} endpoint for environment {environment}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the endpoint

# COMMAND ----------

from databricks.rag.builder_tools import update_endpoint

review_url = update_endpoint(
    environment,
    model_version,
    extended_config,
)

# COMMAND ----------

displayHTML(
    f"Your Review UI is now available. Open the <a href={review_url}>Review UI</a>."
)

# COMMAND ----------

dbutils.notebook.exit(
    f"Your Review UI is now available. Open the Review UI here: {review_url}"
)
