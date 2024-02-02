# Databricks notebook source
# MAGIC %run ./install_libraries

# COMMAND ----------

dbutils.widgets.text("config_root", "", "Configuration root path")
dbutils.widgets.text("config_file", "", "Configuration file path")
dbutils.widgets.text("environment", "", "The current environment")
dbutils.widgets.text("owner", "", "The owner of the environment")

config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")
owner = dbutils.widgets.get("owner")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the environment

# COMMAND ----------

print(f"Setting up environment {environment} using {config_root}/{config_file}")

# COMMAND ----------

from databricks.rag import DefaultExtendedConfig, create_environment

extended_config = DefaultExtendedConfig(config_root, config_file)
create_environment(extended_config.input_config, environment, owner)
print(f"Environment setup complete")

# COMMAND ----------

extended_config = DefaultExtendedConfig(
    config_root, config_file, environment
)
global_config = extended_config.input_config.global_config
environment_info = extended_config.deployed_environment_info
vector_search_endpoint = global_config.vector_search_endpoint
serving_endpoint_name = environment_info.endpoint_name

print(f"Environment configuration {environment_info} set")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Check if the catalog and schema exist

# COMMAND ----------
from databricks.rag.utils import check_if_catalog_and_schema_exist

check_if_catalog_and_schema_exist(extended_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the vector search endpoint, if not already created

# COMMAND ----------

print(f"Setting up vector search endpoint {vector_search_endpoint}")

# COMMAND ----------

from databricks.rag.builder_tools import create_vector_search_endpoint

create_vector_search_endpoint(vector_search_endpoint)

# COMMAND ----------

print(f"Vector search endpoint: {vector_search_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the chain model endpoint, if not already created

# COMMAND ----------

print(f"Setting up chain model endpoint {serving_endpoint_name}")

# COMMAND ----------

from databricks.rag.scoring import FeedbackModel
from databricks.rag.builder_tools import create_endpoint

feedback_model = FeedbackModel(extended_config)
create_endpoint("feedback", feedback_model.name, extended_config)

# COMMAND ----------

print(f"Chain endpoint {serving_endpoint_name}")

# COMMAND ----------

catalog = global_config.uc_assets_location.catalog
schema = global_config.uc_assets_location.schema

from databricks.rag.evaluation.schemas import EVAL_DATASET_INPUT_SCHEMA
from databricks.rag.constants import EVAL_DATASET_TEMPLATE

eval_table_template = spark.createDataFrame([], EVAL_DATASET_INPUT_SCHEMA)

eval_table_template.write.format("delta").mode("overwrite").option(
    "mergeSchema", "true"
).saveAsTable(
    f"{catalog}.{schema}.`rag_studio_{global_config.name}_{environment}_{EVAL_DATASET_TEMPLATE}`"
)

# COMMAND ----------

print("Completed eval dataset template creation")
