# Databricks notebook source
# MAGIC %run ./install_libraries

# COMMAND ----------

# DBTITLE 1,Params
dbutils.widgets.text("config_root", defaultValue="", label="Config root path")
dbutils.widgets.text("config_file", defaultValue="", label="Config file name")
config_root = dbutils.widgets.get("config_root")
config_file = dbutils.widgets.get("config_file")

DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/en/doc-sitemap.xml"
SOURCE_URL_PREFIX = "https://docs.databricks.com/en/"

from databricks.rag import DefaultExtendedConfig
from databricks.rag.builder_tools import create_source_data_uc_volume, store_source_metadata
from my_rag_builder.utils.download_utils import download_html_content
from databricks.rag.utils import save_content

extended_config = DefaultExtendedConfig(
    config_root, config_file
)

# COMMAND ----------

# Create the UC Volume to store the articles
source_uc_volume = create_source_data_uc_volume(extended_config)

# save metadata
source_metadata = {
    "DATABRICKS_SITEMAP_URL": DATABRICKS_SITEMAP_URL,
    "SOURCE_URL_PREFIX": SOURCE_URL_PREFIX,
}
store_source_metadata(extended_config, source_metadata)

html_content_col = "html_content"
doc_uri_col = "doc_uri"

# Download Databricks documentation to a DataFrame
print(f"Downloading Databricks documentation...")
doc_articles_df = download_html_content(
    sitemap_url=DATABRICKS_SITEMAP_URL,
    url_col=doc_uri_col,
    html_content_col=html_content_col,
)

# Upload the articles to the UC Volume
print(f"Saving documents to UC Volume...")
save_content(
    df=doc_articles_df,
    url_prefix=SOURCE_URL_PREFIX,
    uc_volume=source_uc_volume,
    content_column=html_content_col,
    source_uri_column=doc_uri_col,
)

# COMMAND ----------
dbutils.notebook.exit(
    f"Successfully downloaded and uploaded Databricks documentation articles to UC Volume '{source_uc_volume.full_name()}'"
)
