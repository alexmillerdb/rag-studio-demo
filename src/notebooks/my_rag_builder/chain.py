import os

from langchain_community.chat_models import ChatDatabricks
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter
from typing import Callable, Dict, Any, Tuple
from databricks.rag import constants, DefaultExtendedConfig
from databricks.rag.entities import (
    DataProcessor,
    Retriever,
    Chain,
)


# The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]


# The history is everything before the last question
def extract_history(input):
    return input[:-1]


# Method to format the docs returned by the retriever
def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])


def get_retriever_wrapper(
    workspace_url: str,
    catalog_name: str,
    schema_name: str,
    vector_search_endpoint_name: str,
    vector_search_index_name: str,
    data_processor: DataProcessor,
    retriever: Retriever,
) -> Callable[[str], Any]:
    """
    This function returns a retriever that can be used in the chain

    :param workspace_url: The host url of the workspace
    :param catalog_name: The catalog name where the vector search index is located
    :param schema_name: The schema within the catalog where the vector search index is located
    :param vector_search_endpoint_name: The name of the vector search endpoint
    :param vector_search_index_name: The name of the vector search index
    :param data_processor: The data processor configuration used to process the data to create embeddings
    :param retriever: The retriever configuration

    :return: A function which returns back loader_fn of the retriever that can be used in the chain
    """

    embedding_model_endpoint_name = data_processor.embedding_model.endpoint_name
    query_instruction = data_processor.embedding_model.query_instructions
    retriever_params = retriever.configurations

    def get_retriever(persist_dir: str = None):
        from databricks.vector_search.client import VectorSearchClient
        from langchain_community.vectorstores import DatabricksVectorSearch
        from langchain_community.embeddings import DatabricksEmbeddings

        os.environ["DATABRICKS_HOST"] = workspace_url
        # NOTE: your question embedding model used here must match the embedding model used in by indexer in document processing pipeline
        embedding_model = DatabricksEmbeddings(
            endpoint=embedding_model_endpoint_name,
            query_params={"instruction": query_instruction},
        )

        # Get the vector search index
        vsc = VectorSearchClient(
            workspace_url=workspace_url,
            personal_access_token=os.environ.get("DATABRICKS_TOKEN"),
        )
        vs_index = vsc.get_index(
            endpoint_name=vector_search_endpoint_name,
            index_name=f"{catalog_name}.{schema_name}.{vector_search_index_name}",
        )

        # Create the retriever
        vectorstore = DatabricksVectorSearch(
            vs_index,
            text_column=constants.VS_INDEX_TEXT_COL,
            embedding=embedding_model,
            columns=[constants.VS_INDEX_ID_COL, constants.VS_INDEX_DOC_URL_COL],
        )
        return vectorstore.as_retriever(search_kwargs=retriever_params)

    return get_retriever


def get_chat_model(llm_model: str, llm_parameters: Dict[str, Any]) -> ChatDatabricks:
    """
    This function returns a llm chat model that can be used in the chain

    :param llm_model: The endpoint of the llm model
    :param llm_parameters: The parameters for the llm model

    :return: A ChatDatabricks model that can be used in the chain
    """
    return ChatDatabricks(
        endpoint=llm_model,
        **llm_parameters,
    )


def get_chain_builder(
    workspace_url: str,
    catalog_name: str,
    schema_name: str,
    vector_search_endpoint_name: str,
    vector_search_index_name: str,
    data_processor: DataProcessor,
    retriever: Retriever,
    chain_config: Chain,
) -> Tuple[Callable[[str], Any], Any]:
    """
    This function returns a chain that will based on the chat history and the question,
    will generate a query to retrieve relevant documents and later that return
    the answer to the question asked.

    :param workspace_url: The host url of the workspace
    :param catalog_name: The catalog name where the vector search index is located
    :param schema_name: The schema within the catalog where the vector search index is located
    :param vector_search_endpoint_name: The name of the vector search endpoint
    :param vector_search_index_name: The name of the vector search index
    :param data_processor: The data processor configuration used to process the data to create embeddings
    :param retriever: The retriever configuration
    :param chain_config: The chain configuration

    :return: A tuple with function which returns back loader_fn of the retriever that can be used in the chain
    and the full chain
    """
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.prompts import PromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    foundational_model = chain_config.foundational_models[0]
    llm_model = foundational_model.endpoint_name
    llm_parameters = foundational_model.configurations
    llm_prompt_template = foundational_model.prompt_template.template_string
    llm_prompt_template_variables = (
        foundational_model.prompt_template.template_variables
    )
    get_retriever = get_retriever_wrapper(
        workspace_url=workspace_url,
        catalog_name=catalog_name,
        schema_name=schema_name,
        vector_search_endpoint_name=vector_search_endpoint_name,
        vector_search_index_name=vector_search_index_name,
        data_processor=data_processor,
        retriever=retriever,
    )

    generate_query_to_retrieve_context_template = """
    Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

    Chat history: {chat_history}

    Question: {question}
    """

    generate_query_to_retrieve_context_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=generate_query_to_retrieve_context_template,
    )
    question_with_history_and_context_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=llm_prompt_template_variables,
    )
    chat_model = get_chat_model(llm_model, llm_parameters)

    full_chain = (
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
        }
        | RunnablePassthrough()
        | {
            "relevant_docs": generate_query_to_retrieve_context_prompt
            | chat_model
            | StrOutputParser()
            | get_retriever(),
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        }
        | {
            "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        }
        | question_with_history_and_context_prompt
        | chat_model
        | StrOutputParser()
    )
    return get_retriever, full_chain


if __name__ == "__main__":
    # Fill in the following to manually test the code in local environment
    config_root = "../../../config"  # The root path where the dab config is stored
    config_file = (
        "rag-config.yml"  # The path to the dab config file relative to the config_root
    )
    vector_search_index_name = (
        ""  # The name of the vector search index in databricks workspace
    )
    # Set your Databricks personal access token as environment variable: DATABRICKS_TOKEN
    # Example: os.environ["DATABRICKS_TOKEN"] = personal_access_token
    assert (
        "DATABRICKS_TOKEN" in os.environ
    ), "Please set the DATABRICKS_TOKEN environment variable."

    extended_config = DefaultExtendedConfig(config_root, config_file)
    input_config = extended_config.input_config
    global_config = input_config.global_config

    experiment_name = global_config.mlflow_experiment_name
    catalog_name = global_config.uc_assets_location.catalog
    schema_name = global_config.uc_assets_location.schema
    vector_search_endpoint_name = global_config.vector_search_endpoint
    workspace_url = global_config.workspace_url

    chain_config = input_config.chains[0]
    retriever = chain_config.retrievers[0]
    data_processor = retriever.data_processors[0]

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

    retriever = get_retriever()
    retrieve_document_chain = (
        itemgetter("messages") | RunnableLambda(extract_question) | retriever
    )
    # Use the following code for manual testing of the retriever
    #
    # question = "What is Apache Spark?"
    # retriever_response = retrieve_document_chain.invoke(
    #     {"messages": [{"role": "user", "content": question}]}
    # )
    # print("Documents response from the retriever:", retriever_response)

    # import langchain
    # langchain.debug = True #uncomment to see the chain details and the full prompt being sent
    # Use the following code for manual testing of the full chain
    #
    # input_example = {
    #     "messages": [
    #         {"role": "user", "content": "What is Apache Spark?"},
    #         {
    #             "role": "assistant",
    #             "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics.",
    #         },
    #         {"role": "user", "content": "Does it support streaming?"},
    #     ]
    # }
    # chain_response = full_chain.invoke(input_example)
    # print("Final output from the chain:", chain_response)
