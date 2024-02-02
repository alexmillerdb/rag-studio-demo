# Databricks notebook source
# MAGIC %md
# MAGIC ## Install packages required in notebooks
# MAGIC ## (Do not edit!)

# COMMAND ----------

# This notebook allows the custom installation of the RAG library. This is useful for development and testing the library before it is released to the bundle.
dbutils.widgets.text(
    "rag_wheel_path",
    "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/rag-studio/ed24b030-3c87-40b1-b04c-bb1977254aa3/databricks_rag-0.0.0a1-py3-none-any.whl",
    "The wheel path for Databricks RAG library",
)
RAG_WHEEL = dbutils.widgets.get("rag_wheel_path")
# COMMAND ----------

dbutils.widgets.text("task_dependencies", "")
TASK_DEPENDENCIES = dbutils.widgets.get("task_dependencies")

# COMMAND ----------

print(f"Installing libraries {RAG_WHEEL} {TASK_DEPENDENCIES}")

# COMMAND ----------

# MAGIC %pip install $RAG_WHEEL $TASK_DEPENDENCIES

# COMMAND ----------
# MAGIC dbutils.library.restartPython()
