"""Helper functions for ingesting data."""
import xml.etree.ElementTree as ET

import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def download_html_content(
        sitemap_url: str,
        url_col: str,
        html_content_col: str,
        max_documents: int = None,
) -> DataFrame:
    """
    Download databricks documentation articles from databricks doc site and return a Spark DataFrame.

    :param url_col: Column name of the URLs in the result DataFrame.
    :param html_content_col: Column name of the HTML content in the result DataFrame.
    :param max_documents: Maximum number of documents to download; if not set, all documents will be downloaded.
    :return: Spark DataFrame with columns `url_col` and `html_content_col`
    """
    # Fetch the XML content from sitemap
    response = requests.get(sitemap_url)
    root = ET.fromstring(response.content)

    # Find all 'loc' elements (URLs) in the XML
    urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    if max_documents:
        urls = urls[:max_documents]

    def download_content(url):
        """Download HTML content from a URL."""
        try:
            response = requests.get(url)
            # Raise an error if the HTTP request returned an unsuccessful status code
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    download_html_udf = udf(download_content, StringType())

    # Create DataFrame from URLs
    spark = SparkSession.getActiveSession()
    df_urls = spark.createDataFrame(urls, StringType()).toDF(url_col)

    # Apply UDF to DataFrame
    df_with_html = df_urls.withColumn(html_content_col, download_html_udf(url_col))

    # Select and filter non-null results
    final_df = (
        df_with_html
        .select(url_col, html_content_col)
        .filter(f"{html_content_col} IS NOT NULL")
    )
    if final_df.isEmpty():
        raise Exception(
            f"Dataframe is empty, could not download content from {sitemap_url}. Check status."
        )

    return final_df
