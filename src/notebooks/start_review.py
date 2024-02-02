# Databricks notebook source
# MAGIC %run ./install_libraries

# COMMAND ----------

dbutils.widgets.removeAll()

dbutils.widgets.text("config_root", "", "Configuration root path")
dbutils.widgets.text("config_file", "", "Configuration file path")
dbutils.widgets.text("environment", "", "The current environment")
dbutils.widgets.text("version", "", "Model version to deploy")
dbutils.widgets.text("review_request_table", "", "Table with review requests")

config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")
model_version = dbutils.widgets.get("version")
review_request_table = dbutils.widgets.get("review_request_table")

# COMMAND ----------

from databricks.rag import DefaultExtendedConfig

extended_config = DefaultExtendedConfig(
    config_root, config_file, environment
)

# COMMAND ----------

print(
    f"Starting new review workflow using {review_request_table} view for model version {model_version}"
)

# COMMAND ----------

from databricks.rag.review import start_review

review_url = start_review(
    environment,
    model_version,
    review_request_table,
    extended_config,
)
displayHTML(
    f'Your Review UI is now available. Open the <a href="{review_url}">Review UI</a>.'
)

# COMMAND ----------

dbutils.notebook.exit(
    f"Your Review UI is now available. Open the Review UI here: {review_url}"
)
