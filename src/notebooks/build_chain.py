# Databricks notebook source
# MAGIC %run ./install_libraries $task_dependencies="SQLAlchemy==2.0.25"

# COMMAND ----------

import os

CREATE_RAG_VERSION_TASK_NAME = "create_rag_version_task"
PARENT_RUN_ID_KEY = "parent_run_id"
VS_INDEX_NAME_KEY = "vs_index_name"

dbutils.widgets.removeAll()

dbutils.widgets.text("config_root", "", "Configuration root path")
dbutils.widgets.text("config_file", "", "Configuration file path")
dbutils.widgets.text("environment", "", "The current environment")

config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")
parent_run_id = dbutils.jobs.taskValues.get(
    taskKey=CREATE_RAG_VERSION_TASK_NAME, key=PARENT_RUN_ID_KEY
)
vector_search_index_name = dbutils.jobs.taskValues.get(
    taskKey=CREATE_RAG_VERSION_TASK_NAME,
    key=VS_INDEX_NAME_KEY,
)

# COMMAND ----------

import dataclasses

from databricks.rag import DefaultExtendedConfig

extended_config = DefaultExtendedConfig(config_root, config_file, environment)
input_config = extended_config.input_config
global_config = input_config.global_config
environment_info = extended_config.deployed_environment_info

experiment_name = global_config.mlflow_experiment_name
catalog_name = global_config.uc_assets_location.catalog
schema_name = global_config.uc_assets_location.schema
workspace_url = global_config.workspace_url
vector_search_endpoint_name = global_config.vector_search_endpoint

model_name = environment_info.model_name
serving_endpoint_name = environment_info.endpoint_name
security_scope = environment_info.security_scope
security_key = environment_info.security_key
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(security_scope, security_key)

chain_config = input_config.chains[0]
retriever = chain_config.retrievers[0]
data_processor = retriever.data_processors[0]

# COMMAND ----------

print(f"Setting up chain model {model_name} with {environment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building the chain model

# COMMAND ----------
from my_rag_builder.chain import get_chain_builder

get_retriever, full_chain = get_chain_builder(
    workspace_url=workspace_url,
    catalog_name=catalog_name,
    schema_name=schema_name,
    vector_search_endpoint_name=vector_search_endpoint_name,
    vector_search_index_name=vector_search_index_name,
    data_processor=data_processor,
    retriever=retriever,
    chain_config=chain_config,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register a model

# COMMAND ----------

from databricks.rag.builder_tools import log_register_chain_model


# This code is required for serving the chain so it can get my_rag_builder as dependencies. Do not edit!
def code_path():
    import os
    from pathlib import Path

    full_path = Path(
        "/Workspace"
        + str(
            Path(
                dbutils.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .notebookPath()
                .get()
            ).parent
        )
    )
    return full_path / "my_rag_builder"


registered_model = log_register_chain_model(
    extended_config,
    parent_run_id,
    get_retriever,
    full_chain,
    dataclasses.asdict(chain_config),
    [code_path()],
)

# COMMAND ----------

# Use the following code for manual testing of the mlflow logged model
#
# model = mlflow.langchain.load_model(registered_model.model_info.model_uri)
# model.invoke(input_example)

# COMMAND ----------

dbutils.jobs.taskValues.set(
    key="model_version", value=registered_model.model_version.version
)

# COMMAND ----------

dbutils.notebook.exit(
    f"Created chain {model_name} model with version {registered_model.model_version.version}"
)
