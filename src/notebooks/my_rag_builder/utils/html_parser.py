"""Helper functions for parsing HTML content and chunking."""
from typing import List

import pyspark.sql.functions as F
from bs4 import BeautifulSoup
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from transformers import AutoTokenizer


def parse_and_chunk_html_content(
        df_with_html: DataFrame,
        html_content_col: str,
        chunk_text_col: str,
        max_chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int = 20,
) -> DataFrame:
    """
    Parse the raw text from the databricks documentation and split it into chunks.

    Note: the output DataFrame does NOT contain the `html_content_col` from the input DataFrame.

    :param df_with_html: Spark DataFrame that contains the raw HTML content.
    :param html_content_col: Column name of the HTML content in the input DataFrame.
    :param chunk_text_col: Column name of the chunk text in the result DataFrame.
    :param max_chunk_size: Maximum number of tokens in a chunk.
    :param chunk_overlap: Number of tokens to overlap between chunks.
    :param min_chunk_size: Minimum size of a chunk. Chunks with fewer tokens will be discarded.
    :return: Spark DataFrame that contains chunk text.
    """
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=max_chunk_size, chunk_overlap=chunk_overlap
    )
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

    def split_html_on_h2(html: str) -> List[str]:
        """Split HTML content on H2 headers. Merge small h2 chunks together. Discard too small chunks."""
        # Parse HTML to get the article body
        soup = BeautifulSoup(html, "html.parser")
        article_div = soup.find("div", itemprop="articleBody")
        if article_div:
            article_body = str(article_div).strip()
        else:
            return []
        if not article_body:
            return []
        # Split on H2
        h2_chunks = html_splitter.split_text(article_body)
        chunks = []
        previous_chunk = ""
        # Merge chunks together to avoid too small docs.
        for c in h2_chunks:
            # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
            content = c.metadata.get("header2", "") + "\n" + c.page_content
            if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size / 2:
                previous_chunk += content + "\n"
            else:
                chunks.extend(text_splitter.split_text(previous_chunk.strip()))
                previous_chunk = content + "\n"
        if previous_chunk:
            chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        # Discard too small chunks
        return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]

    parse_and_split_udf = udf(split_html_on_h2, ArrayType(StringType()))

    return (df_with_html
            .withColumn(chunk_text_col, F.explode(parse_and_split_udf(html_content_col)))
            .drop(html_content_col))
