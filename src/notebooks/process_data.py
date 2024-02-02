# Databricks notebook source
# MAGIC %run ./install_libraries $task_dependencies="lxml>=4.9.3 transformers"

# COMMAND ----------

# DBTITLE 1,Params
CREATE_RAG_VERSION_TASK_NAME = "create_rag_version_task"
PARENT_RUN_ID_KEY = "parent_run_id"
VS_INDEX_NAME_KEY = "vs_index_name"
EMBEDDINGS_TABLE_NAME_KEY = "embeddings_table_name"

parent_run_id = dbutils.jobs.taskValues.get(
    taskKey=CREATE_RAG_VERSION_TASK_NAME,
    key=PARENT_RUN_ID_KEY,
)
vector_search_index_name = dbutils.jobs.taskValues.get(
    taskKey=CREATE_RAG_VERSION_TASK_NAME,
    key=VS_INDEX_NAME_KEY,
)
embeddings_table_name = dbutils.jobs.taskValues.get(
    taskKey=CREATE_RAG_VERSION_TASK_NAME,
    key=EMBEDDINGS_TABLE_NAME_KEY,
)

dbutils.widgets.text("config_root", defaultValue="", label="Config root path")
dbutils.widgets.text("config_file", defaultValue="", label="Config file name")
dbutils.widgets.text("environment", defaultValue="", label="Current environment")
config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")
environment = dbutils.widgets.get("environment")

from databricks.rag import DefaultExtendedConfig
from databricks.rag.builder_tools import get_source_data_uc_volume, get_source_metadata
from databricks.rag.entities import (
    UnityCatalogTable,
    UnityCatalogVectorSearchIndex,
)
from databricks.rag.builder_tools.data_processing import write_embeddings_to_table
from my_rag_builder import document_processor

extended_config = DefaultExtendedConfig(
    config_root, config_file, environment
)
global_config = extended_config.input_config.global_config
data_processor_config = extended_config.input_config.data_processors[0]
destination_table_config = data_processor_config.destination_table
experiment_name = global_config.mlflow_experiment_name

catalog = global_config.uc_assets_location.catalog
schema = global_config.uc_assets_location.schema
vector_search_endpoint = global_config.vector_search_endpoint
chunk_size = int(
    data_processor_config.get_configurations().get("chunk_size", 500)
)
chunk_overlap = int(
    data_processor_config.get_configurations().get("chunk_overlap", 50)
)
embedding_model_endpoint = data_processor_config.embedding_model.endpoint_name
embedding_instruction = data_processor_config.embedding_model.embedding_instructions
source_metadata = get_source_metadata(extended_config)
source_data_uc_volume = get_source_data_uc_volume(extended_config)
vector_search_index = UnityCatalogVectorSearchIndex(
    catalog_name=catalog, schema_name=schema, index_name=vector_search_index_name
)
embeddings_table = UnityCatalogTable(
    catalog_name=catalog,
    schema_name=schema,
    table_name=embeddings_table_name,
)
# COMMAND ----------

# DBTITLE 1,Start MLflow child run
import mlflow

mlflow.set_experiment(experiment_name)
mlflow.start_run(run_id=parent_run_id)
mlflow.start_run(nested=True)
run = mlflow.active_run()
run_id = run.info.run_id
print(f"Active run_id: {run_id}; Parent run_id: {parent_run_id}")

# COMMAND ----------
embeddings_data = document_processor.run(
    source_metadata=source_metadata,
    source_data_volume=source_data_uc_volume,
    embedding_model_endpoint=embedding_model_endpoint,
    embedding_instruction=embedding_instruction,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Write to embeddings table
write_embeddings_to_table(
    embeddings_df=embeddings_data.df,
    embeddings_table=embeddings_table,
    chunk_id_col=embeddings_data.chunk_id_col,
    doc_uri_col=embeddings_data.doc_uri_col,
    text_col=embeddings_data.chunk_text_col,
    embedding_col=embeddings_data.embedding_col,
)

# Trigger a sync to the vector search index
from databricks.vector_search.client import VectorSearchClient
from databricks.rag.builder_tools import (
    wait_for_vs_endpoint_to_be_ready,
    wait_for_index_to_be_ready,
)

vs_client = VectorSearchClient()

# Wait for the vector search endpoint to be ready
print(f"Waiting for vector search endpoint {vector_search_endpoint} to be ready...")
wait_for_vs_endpoint_to_be_ready(vs_client, vector_search_endpoint)

# Trigger a sync to update vector search index
print(f"Syncing index {vector_search_index} on endpoint {vector_search_endpoint}...")
wait_for_index_to_be_ready(
    vs_client, vector_search_endpoint, vector_search_index.full_name(escape=False)
)
vs_client.get_index(
    vector_search_endpoint, vector_search_index.full_name(escape=False)
).sync()

# COMMAND ----------

# DBTITLE 1,End MLflow child run
mlflow.end_run()
mlflow.end_run()

# COMMAND ----------
