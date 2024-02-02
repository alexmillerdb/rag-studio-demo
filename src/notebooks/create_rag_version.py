# Databricks notebook source
# MAGIC %run ./install_libraries

# COMMAND ----------

dbutils.widgets.text("config_root", "", "Configuration root path")
dbutils.widgets.text("config_file", "", "Configuration file path")
dbutils.widgets.text("environment", "", "The current environment")

config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")

# COMMAND ----------

print(
    f"Setting up a new RAG version with {environment} using {config_root}/{config_file}"
)

# COMMAND ----------

from databricks.rag import DefaultExtendedConfig

extended_config = DefaultExtendedConfig(
    config_root, config_file, environment
)

# COMMAND ----------

from databricks.rag.builder_tools import create_rag_version

version_id = create_rag_version(
    environment, config_root, config_file, extended_config
)

# COMMAND ----------

from databricks.rag.builder_tools import create_embeddings_table, create_vector_search_index

embeddings_table = create_embeddings_table(extended_config, version_id)
vector_search_index = create_vector_search_index(extended_config, version_id, embeddings_table)

# COMMAND ----------

dbutils.jobs.taskValues.set(key="parent_run_id", value=version_id)
dbutils.jobs.taskValues.set(key="vs_index_name", value=vector_search_index.index_name)
dbutils.jobs.taskValues.set(key="embeddings_table_name", value=embeddings_table.table_name)
