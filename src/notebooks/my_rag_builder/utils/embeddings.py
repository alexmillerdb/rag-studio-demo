"""Helper functions for embedding."""
import mlflow.deployments
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf


def generate_embeddings(
        df_with_chunks: DataFrame,
        embedding_model_endpoint: str,
        text_col: str,
        embedding_col: str,
        batch_size: int = 150,
        instruction: str = None,
) -> DataFrame:
    """
    Generate embeddings for the chunks.

    :param df_with_chunks: Spark DataFrame with columns `VS_INDEX_DOC_URL_COL` and `VS_INDEX_TEXT_COL`
    :param embedding_model_endpoint: The endpoint name of the embedding model.
    :param text_col: Column name of the text in the input DataFrame.
    :param embedding_col: Column name of the embeddings in the result DataFrame.
    :param batch_size: The batch size to use when generating embeddings.

    :return: Spark DataFrame with columns `VS_INDEX_DOC_URL_COL`, `VS_INDEX_TEXT_COL`, and `VS_INDEX_EMBEDDING_COL`
    """
    # Create a UDF to generate embeddings for the chunks
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    @pandas_udf("array<float>")
    def get_embeddings_udf(contents: pd.Series) -> pd.Series:
        def get_embeddings(batch):
            # Note: this will gracefully fail if an exception is thrown during embedding creation (add try/except if needed)
            inputs = {"input": batch}
            if instruction:
                inputs["instruction"] = instruction
            response = deploy_client.predict(endpoint=embedding_model_endpoint, inputs=inputs)
            return [e["embedding"] for e in response.data]

        # Splitting the contents into batches
        batches = [
            contents.iloc[i: i + batch_size]
            for i in range(0, len(contents), batch_size)
        ]

        # Process each batch and collect the results
        all_embeddings = []
        for batch in batches:
            all_embeddings += get_embeddings(batch.tolist())

        return pd.Series(all_embeddings)

    return df_with_chunks.withColumn(embedding_col, get_embeddings_udf(text_col))
