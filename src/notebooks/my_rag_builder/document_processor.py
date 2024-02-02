from __future__ import annotations

import uuid
from dataclasses import dataclass

from databricks.rag.entities import (
    UnityCatalogVolume,
)
from databricks.rag.utils import load_content
from .utils.embeddings import generate_embeddings
from .utils.html_parser import parse_and_chunk_html_content


@dataclass
class Embeddings:
    df: DataFrame  # noqa: F821
    doc_uri_col: str
    chunk_text_col: str
    embedding_col: str
    chunk_id_col: str


def generate_uuid():
    return str(uuid.uuid4())


def run(
        source_metadata: dict,
        source_data_volume: UnityCatalogVolume,
        embedding_model_endpoint: str,
        embedding_instruction: str,
        chunk_size: int,
        chunk_overlap: int,
) -> Embeddings:
    """
    Load the source data from a UC Volume
    Process the data, chunking, and generate embedding.

    The returned DataFrame must have at least these columns:
    - chunk_id_col: string
    - doc_uri_col: string
    - chunk_text_col: string
    - embedding_col: array<float>

    :param source_metadata: The metadata of the source data.
    :param source_data_volume: The source data UC Volume.
    :param embedding_model_endpoint: The endpoint name of the embedding model.
    :param embedding_instruction: The instruction to use when generating embeddings.
    :param chunk_size: The maximum size of each chunk.
    :param chunk_overlap: The overlap between chunks.

    :return: An EmbeddingTable object contains the DataFrame and column names.
    """
    source_url_prefix = source_metadata.get("SOURCE_URL_PREFIX", "")

    html_content_col = "html_content"
    chunk_id_col = "chunk_id"
    doc_uri_col = "doc_uri"
    chunk_text_col = "text"
    embedding_col = "embedding"

    # Load source documents from UC Volume
    print(f"Loading source documents from UC Volume...")
    docs_df = load_content(
        uc_volume=source_data_volume,
        url_prefix=source_url_prefix,
        content_column=html_content_col,
        source_uri_column=doc_uri_col,
    )
    # Parse and split the documents into chunks
    print(f"Parsing and splitting the articles into chunks...")
    chunks_df = parse_and_chunk_html_content(
        df_with_html=docs_df,
        html_content_col=html_content_col,
        chunk_text_col=chunk_text_col,
        max_chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Generate embeddings for the chunks
    print(f"Generating embeddings for the chunks...")
    chunk_with_embeddings_df = generate_embeddings(
        df_with_chunks=chunks_df,
        embedding_model_endpoint=embedding_model_endpoint,
        text_col=chunk_text_col,
        embedding_col=embedding_col,
        instruction=embedding_instruction,
    )
    from pyspark.sql.functions import udf
    result_df = chunk_with_embeddings_df.withColumn(chunk_id_col, udf(generate_uuid)())

    return Embeddings(
        df=result_df,
        chunk_id_col=chunk_id_col,
        doc_uri_col=doc_uri_col,
        chunk_text_col=chunk_text_col,
        embedding_col=embedding_col,
    )


if __name__ == '__main__':
    # Put in your local testing params and run this file for local testing
    embeddings = run(
        source_metadata={"SOURCE_URL_PREFIX": "https://docs.databricks.com/en/"},
        source_data_volume=UnityCatalogVolume(catalog_name="", schema_name="", volume_name=""),
        embedding_model_endpoint="databricks-bge-large-en",
        embedding_instruction="",
        chunk_size=500,
        chunk_overlap=50,
    )
    # Use the following code to inspect computed embeddings
    # embeddings.df.show(5)
