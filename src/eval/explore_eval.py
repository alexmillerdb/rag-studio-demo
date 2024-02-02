# Databricks notebook source
# MAGIC %md # RAG Evaluation Analysis Notebook
# MAGIC
# MAGIC #### About this notebook
# MAGIC In this notebook, we investigate the offline performance across various RAG versions. We look at a few high level questions:
# MAGIC - How performant is the model, in terms of speed and cost?
# MAGIC - How relevant are the retrievals?
# MAGIC - How are the models generations?
# MAGIC   - How harmful are they?
# MAGIC   - How relevant are they to the question and given context?
# MAGIC   - How faithful are they to the given context?
# MAGIC
# MAGIC #### How to run the notebook
# MAGIC You should populate the parameters at the top with the relevant information. You can then run all the cells and explore the visualizations throughout the notebook. Alternatively, you can run this notebook through the RAG CLI via `./rag explore-eval -env <YOUR_ENV> -eval-table-name <YOUR_TABLE> -v *`. Changing versions should automatically update the visualizations, but changing the table names will require running through the notebook again.
# MAGIC
# MAGIC #### How to customize this notebook
# MAGIC This notebook is meant to be altered for different use cases. You should clone this notebook and make the desired changes to the queries. Then replace the original notebook in `src/eval` with the updated notebook. You can now rerun this notebook through the RAG CLI.

# COMMAND ----------

# MAGIC %md ### Imports & Initializations

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go

from pyspark.sql import functions as F

# COMMAND ----------

# Helper to delimit the table names to allow for special characters
delimit_table_name = lambda x: f'`{"`.`".join(x.split("."))}`'

dbutils.widgets.text("request_log_table_name", "", label="Name of the request log table")
request_log_table_name = delimit_table_name(dbutils.widgets.get("request_log_table_name"))

dbutils.widgets.text("assessment_log_table_name", "", label="Name of the assessment log table")
assessment_log_table_name = delimit_table_name(dbutils.widgets.get("assessment_log_table_name"))

# COMMAND ----------

# Helper to get version number from RAG version. 
# We assume the version number is the last portion of the model URI after the last '/'.
get_rag_version_number = F.udf(lambda x: x.split("/")[-1])
# Helper to transform a list of strings into a SQL list
get_sql_string_list = lambda x: '"' + '","'.join(x) + '"'
# Helper to get color scale for graphs
color_scale = px.colors.sequential.Viridis
get_color_scale = lambda x: str(color_scale[int(x) % len(color_scale)])

# Retrieve all the RAG versions from a scan of the table.
request_log_df = spark.read.table(request_log_table_name)
rag_versions = list(sorted(map(lambda x: x["app_version_id"], request_log_df.select(F.col("trace.app_version_id")).distinct().collect())))
dbutils.widgets.multiselect("versions", "*", choices=["*"] + list(sorted(rag_versions)), label="RAG Version to Visualize")

# COMMAND ----------

# MAGIC %md
# MAGIC ## "Vibe check": Explore a few traces across different RAG versions
# MAGIC In this section, we display traces for several request ids across all RAG versions.

# COMMAND ----------

# Get the specified models from the widget and display the filtered performance metrics table
# Note: this line should be included in every cell such that when the widget is updated, the visualiations are too.
models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
  models_to_compare if models_to_compare != ["*"] else rag_versions)

samples = spark.sql(f"""
    -- Get the distinct request ids from the request log relevant to the specified RAG versions
    WITH REQUEST_IDS AS (
      SELECT DISTINCT request.request_id
      FROM { request_log_table_name }
      WHERE trace.app_version_id in ({models_to_compare_str})
      LIMIT 10
    ),
    -- Get the respective samples
    samples AS (
      SELECT
        request,
        trace,
        trace.app_version_id
      FROM { request_log_table_name }
      WHERE request.request_id IN ( SELECT request_id FROM REQUEST_IDS )
    )
    SELECT *
    FROM
      -- Pivot the table such that we can easily compare across RAG versions
      samples PIVOT (
        FIRST(trace) FOR app_version_id IN ({models_to_compare_str})
      )
""")
display(samples)

# COMMAND ----------

# MAGIC %md
# MAGIC ## How performant is the model and what is the cost?
# MAGIC In this section, we explore the average latency of the model to compare performance and the average number of output tokens to compare the cost.

# COMMAND ----------

models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

performance_metrics_df = spark.sql(
    f"""
  SELECT 
      request_log.request.request_id as request_id,
      assessment_log.step_id,
      request_log.trace.app_version_id as rag_version,
      text_assessment.ratings.latency.double_value as latency,
      text_assessment.ratings.token_count.double_value as token_count
  FROM (
    SELECT *
    FROM {assessment_log_table_name}
    WHERE 
      text_assessment.ratings.latency.double_value IS NOT NULL 
      OR text_assessment.ratings.token_count.double_value IS NOT NULL 
  ) assessment_log 
  JOIN {request_log_table_name} request_log 
  ON request_log.request.request_id = assessment_log.request_id
  WHERE request_log.trace.app_version_id IN ({models_to_compare_str})
"""
)

# Calculate the average number of tokens/latency by RAG version
performance_metrics_df = performance_metrics_df.withColumn("rag_version", get_rag_version_number(F.col("rag_version")))
avg_tokens_df = (performance_metrics_df.groupby("rag_version").avg("token_count").toPandas())
avg_latency_df = performance_metrics_df.groupby("rag_version").avg("latency").toPandas()

performance_fig = sp.make_subplots(rows=1, cols=2)

# Add the bar chart of average number of tokens to the subplot
token_bar = go.Bar(
    y=avg_tokens_df["rag_version"],
    x=avg_tokens_df["avg(token_count)"],
    orientation="h",
    name="# of Tokens",
    marker=dict(color=avg_tokens_df["rag_version"].apply(get_color_scale)),
)
performance_fig.add_trace(token_bar, row=1, col=1)

# Add the bar chart of average latency to the subplot
latency_bar = go.Bar(
    y=avg_latency_df["rag_version"],
    x=avg_latency_df["avg(latency)"],
    orientation="h",
    name="Latency (ms)",
    marker=dict(color=avg_tokens_df["rag_version"].apply(get_color_scale)),
)
performance_fig.add_trace(latency_bar, row=1, col=2)

performance_fig.update_traces(marker_line_color = get_color_scale(0), marker_line_width = 2)
performance_fig.update_layout(
    title={
        "text": "Performance Metrics Across RAG Versions",
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis=dict(title="Average Number of Tokens", side="bottom", domain=[0, 0.48]),
    xaxis2=dict(title="Average Latency (milliseconds)", side="bottom", domain=[0.52, 1]),
    yaxis=dict(title="RAG Version"),
    showlegend=False,
)

performance_fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How relevant are my model retrievals?
# MAGIC In this section, we investigate the relevance of the retrievals from the selected RAG versions. The current supported metrics are precision, recall, and NDCG.

# COMMAND ----------

models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

retrieval_log_df = spark.sql(
    f"""
    WITH RATINGS_TABLE AS (
      SELECT 
        request_log.request.request_id as request_id,
        assessment_log.step_id,
        request_log.trace.app_version_id as rag_version,
        source.type as assessment_source,   
        retrieval_assessment.ratings as ratings
      FROM {assessment_log_table_name} assessment_log 
      JOIN {request_log_table_name} request_log 
      ON request_log.request.request_id = assessment_log.request_id
      WHERE request_log.trace.app_version_id IN ({models_to_compare_str}) 
    ),
    METRICS_TABLE AS (
      -- Get the recall/precision/NDCG values for k=1
      SELECT 
          *, 
          ratings.retrieval_recall_at_1.double_value as retrieval_recall_at_k_score,
          ratings.retrieval_precision_at_1.double_value as retrieval_precision_at_k_score,
          ratings.retrieval_ndcg_at_1.double_value as retrieval_ndcg_at_k_score,
          1 as k
      FROM RATINGS_TABLE
      UNION ALL
      -- Get the recall/precision/NDCG values for k=3
      SELECT
          *,
          ratings.retrieval_recall_at_3.double_value as retrieval_recall_at_k_score,
          ratings.retrieval_precision_at_3.double_value as retrieval_precision_at_k_score,
          ratings.retrieval_ndcg_at_3.double_value as retrieval_ndcg_at_k_score,
          3 as k
      FROM RATINGS_TABLE
      UNION ALL
      -- Get the recall/precision/NDCG values for k=5
      SELECT
          *,
          ratings.retrieval_recall_at_5.double_value as retrieval_recall_at_k_score,
          ratings.retrieval_precision_at_5.double_value as retrieval_precision_at_k_score,
          ratings.retrieval_ndcg_at_5.double_value as retrieval_ndcg_at_k_score,
          5 as k
      FROM RATINGS_TABLE
    )
    SELECT *
    FROM METRICS_TABLE
    -- Drop any rows for which all three metrics are null
    WHERE retrieval_recall_at_k_score IS NOT NULL 
      OR retrieval_precision_at_k_score IS NOT NULL
      OR retrieval_ndcg_at_k_score IS NOT NULL
"""
)

if retrieval_log_df.count():
    k_values = [1, 3, 5]
    retrieval_log_df = retrieval_log_df.withColumn("rag_version_number", get_rag_version_number(F.col("rag_version")))

    retrieval_fig = sp.make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Recall@k by RAG Version",
            "Precision@k by RAG Version",
            "NDCG@k by RAG Version",
        ),
    )

    # Plot recall metrics
    avg_recall_df = (
        retrieval_log_df.groupby(["k", "rag_version_number"])
        .mean("retrieval_recall_at_k_score")
        .toPandas()
    )
    for k in k_values:
        recall_df_k = avg_recall_df[avg_recall_df["k"] == k]
        retrieval_fig.add_trace(
            go.Bar(
                x=recall_df_k["avg(retrieval_recall_at_k_score)"],
                y=recall_df_k["rag_version_number"],
                orientation="h",
                name=f"Recall@{k}",
                text=f"k={k}",
                marker=dict(
                    color=recall_df_k["rag_version_number"]
                    .apply(lambda x: get_color_scale(int(x * 3)))
                    .unique(),
                    opacity=recall_df_k["avg(retrieval_recall_at_k_score)"],
                ),
            ),
            row=1,
            col=1,
        )

    # Plot precision metrics
    avg_precision_df = (
        retrieval_log_df.groupby(["k", "rag_version_number"])
        .mean("retrieval_precision_at_k_score")
        .toPandas()
    )
    for k in k_values:
        precision_df_k = avg_precision_df[avg_precision_df["k"] == k]
        retrieval_fig.add_trace(
            go.Bar(
                x=precision_df_k["avg(retrieval_precision_at_k_score)"],
                y=precision_df_k["rag_version_number"],
                orientation="h",
                name=f"Precision@{k}",
                text=f"k={k}",
                marker=dict(
                    color=recall_df_k["rag_version_number"].apply(
                        lambda x: get_color_scale(str(int(x * 3)))
                    ),
                    opacity=precision_df_k["avg(retrieval_precision_at_k_score)"],
                ),
            ),
            row=1,
            col=2,
        )

    # Plot NDCG metrics
    avg_ndcg_df = (
        retrieval_log_df.groupby(["k", "rag_version_number"])
        .mean("retrieval_ndcg_at_k_score")
        .toPandas()
    )
    for k in k_values:
        ndcg_df_k = avg_ndcg_df[avg_ndcg_df["k"] == k]
        retrieval_fig.add_trace(
            go.Bar(
                x=ndcg_df_k["avg(retrieval_ndcg_at_k_score)"],
                y=ndcg_df_k["rag_version_number"],
                orientation="h",
                name=f"NDCG@{k}",
                text=f"k={k}",
                marker=dict(
                    color=recall_df_k["rag_version_number"].apply(
                        lambda x: get_color_scale(str(int(x * 3)))
                    ),
                    opacity=ndcg_df_k["avg(retrieval_ndcg_at_k_score)"],
                ),
            ),
            row=1,
            col=3,
        )

    # Update layout
    retrieval_fig.update_traces(marker_line_color = get_color_scale(0), marker_line_width = 2)
    retrieval_fig.update_layout(
        title={
            "text": "Retrieval Metrics Across RAG Versions",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis=dict(
            title="Average Recall@k Score",
            side="bottom",
        ),
        xaxis2=dict(
            title="Average Precision@k Score",
            side="bottom",
        ),
        xaxis3=dict(
            title="Average NDCG@k Score",
            side="bottom",
        ),
        yaxis=dict(title="RAG Version")
    )

    # Show the plot
    retrieval_fig.show()

else:
    print(
        "No retrieval metrics to display. This happens when the input data lacks ground-truth for the retrieval steps."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## How are my model's generations?
# MAGIC In this section, we investigate the generations of the selected RAG versions. We'll look at metrics like harmfulness, faithfulness to the context, and more.

# COMMAND ----------

# MAGIC %md
# MAGIC #### How harmful are my generations?

# COMMAND ----------

# DBTITLE 1,Name: Harmfulness Log Analysis with Model Comparison
models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

harmfulness_log_df = spark.sql(
    f"""
    SELECT 
        request_log.request.request_id as request_id,
        assessment_log.step_id,
        request_log.trace.app_version_id as rag_version,
        source.type as assessment_source,
        request_log.output as output_message,
        CAST(text_assessment.ratings.harmful.bool_value AS INTEGER) as harmful_value,
        text_assessment.ratings.harmful.double_value as harmful_score
    FROM (
      SELECT *
      FROM {assessment_log_table_name}
      WHERE 
        text_assessment.ratings.harmful.bool_value IS NOT NULL
        OR text_assessment.ratings.harmful.double_value IS NOT NULL
    ) assessment_log 
    JOIN {request_log_table_name} request_log 
    ON request_log.request.request_id = assessment_log.request_id
    WHERE request_log.trace.app_version_id IN ({models_to_compare_str})
"""
)

if harmfulness_log_df.count():
    # Plot the relevance metrics across RAG versions
    harmfulness_fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Percentage of Harmful Generations",
            "Box Plot of Harmfulness Scores",
        ),
    )
    harmfulness_log_df = harmfulness_log_df.withColumn("rag_version_number", get_rag_version_number(F.col("rag_version")))
    harmfulness_log_pdf = harmfulness_log_df.toPandas()

    # Bar chart of average harmfulness
    avg_harmfulness_pdf = harmfulness_log_pdf.groupby(["rag_version_number"]).mean().reset_index()
    harmfulness_bar = go.Bar(
        x=avg_harmfulness_pdf["harmful_value"],
        y=avg_harmfulness_pdf["rag_version_number"],
        orientation="h",
        name="Percentage of Harmful Generations",
        marker=dict(color=avg_harmfulness_pdf["rag_version_number"].apply(get_color_scale).unique()),
    )
    harmfulness_fig.add_trace(harmfulness_bar, row=1, col=1)
    harmfulness_fig.update_traces(marker_line_color = get_color_scale(0), marker_line_width = 2)

    # Box plot of harmfulness
    for rag_version_number in list(avg_harmfulness_pdf["rag_version_number"].unique()):
        harmfulness_box = go.Box(
            x=harmfulness_log_pdf[harmfulness_log_pdf["rag_version_number"] == rag_version_number]["harmful_score"],
            y=harmfulness_log_pdf[harmfulness_log_pdf["rag_version_number"] == rag_version_number]["rag_version_number"],
            orientation="h",
            name="Harmfulness Score",
            marker=dict(color=get_color_scale(rag_version_number)),
        )
        harmfulness_fig.append_trace(harmfulness_box, row=1, col=2)

    # Update layout
    harmfulness_fig.update_layout(
        title={
            "text": "Harmfulness Metrics Across RAG Versions",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis=dict(
            title="Percentage Harmful",
            side="bottom",
            domain=[0, 0.48],
        ),
        xaxis2=dict(
            title="Harmfulness Score",
            side="bottom",
            domain=[0.52, 1],
        ),
        yaxis=dict(title="RAG Version"),
        showlegend=False,
    )

    # Show the plot
    harmfulness_fig.show()
else:
    print("No harmfulness results to display.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### What are the most harmful generations by model?

# COMMAND ----------

# DBTITLE 1,Top K Harmfulness Examples by Version
models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

harmfulness_log_df.createOrReplaceTempView("harmful_df")
k = 5
top_k_harmful_examples_df = spark.sql(
    f"""
      WITH RankedHarmfulness AS (
        SELECT 
            *,
            -- Rank the harmfulness scores to get the top k examples
            ROW_NUMBER() OVER (
                PARTITION BY rag_version
                ORDER BY harmful_score ASC
            ) AS row_num
        FROM harmful_df
      )
      SELECT 
          request_id,
          step_id,
          rag_version,
          harmful_score,
          output_message
      FROM RankedHarmfulness
      WHERE row_num <= {k}
        AND rag_version in ({models_to_compare_str})
    """
)

display(top_k_harmful_examples_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Are my generations relevant to the question and given context?

# COMMAND ----------

# DBTITLE 1,Spark Relevance Log Data Extraction
models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

relevance_log_df = spark.sql(
    f"""
    SELECT 
        request_log.request.request_id as request_id,
        assessment_log.step_id,
        request_log.trace.app_version_id as rag_version,
        source.type as assessment_source,
        request_log.output as output_message,
        CAST(text_assessment.ratings.relevant_to_question_and_context.bool_value AS INTEGER) as relevant_to_question_and_context_value,
        text_assessment.ratings.relevant_to_question_and_context.double_value as relevant_to_question_and_context_score
    FROM (
      SELECT *
      FROM {assessment_log_table_name}
      WHERE text_assessment.ratings.relevant_to_question_and_context.bool_value IS NOT NULL
    ) assessment_log 
    JOIN {request_log_table_name} request_log 
    ON request_log.request.request_id = assessment_log.request_id
    WHERE request_log.trace.app_version_id IN ({models_to_compare_str})
"""
)

if relevance_log_df.count():
    # Plot the relevance metrics across RAG versions
    relevance_fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Percentage of Relevant Generations (to Question and Context)",
            "Box Plot of Relevance Scores (to Question and Context)",
        ),
    )
    relevance_log_df = relevance_log_df.withColumn("rag_version_number", get_rag_version_number(F.col("rag_version")))
    relevance_log_pdf = relevance_log_df.toPandas()

    # Bar chart of average relevance to question and context
    avg_relevance_pdf = relevance_log_pdf.groupby(["rag_version_number"]).mean().reset_index()
    relevance_bar = go.Bar(
        x=avg_relevance_pdf["relevant_to_question_and_context_value"],
        y=avg_relevance_pdf["rag_version_number"],
        orientation="h",
        name="Percentage of Relevant Generations",
        marker=dict(color=avg_relevance_pdf["rag_version_number"].apply(get_color_scale).unique()),
    )
    relevance_fig.add_trace(relevance_bar, row=1, col=1)
    relevance_fig.update_traces(marker_line_color = get_color_scale(0), marker_line_width = 2)

    # Box plot of relevance to question to context
    for rag_version_number in list(avg_relevance_pdf["rag_version_number"].unique()):
        relevance_box = go.Box(
            x=relevance_log_pdf[relevance_log_pdf["rag_version_number"] == rag_version_number]["relevant_to_question_and_context_score"],
            y=relevance_log_pdf[relevance_log_pdf["rag_version_number"] == rag_version_number]["rag_version_number"],
            orientation="h",
            name="Relevance Score (to Question and Context)",
            marker=dict(color=get_color_scale(rag_version_number)),
        )
        relevance_fig.append_trace(relevance_box, row=1, col=2)

    # Update layout
    relevance_fig.update_layout(
        title={
            "text": "Relevance Metrics Across RAG Versions",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis=dict(
            title="Percentage Relevant to Question and Context",
            side="bottom",
            domain=[0, 0.48],
        ),
        xaxis2=dict(
            title="Relevance Score (to Question and Context)",
            side="bottom",
            domain=[0.52, 1],
        ),
        yaxis=dict(title="RAG Version"),
        showlegend=False,
    )

    # Show the plot
    relevance_fig.show()
else:
    print("No relevance results to display.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### What are the least relevant generations?

# COMMAND ----------

# DBTITLE 1,Rank and Display Bottom k Relevant Examples
models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

relevance_log_df.createOrReplaceTempView("relevance_df")
k = 5
bottom_k_relevant_examples_df = spark.sql(
    f"""
      WITH RankedRelevance AS (
        SELECT 
            *,
            -- Rank the relevance scores to get the bottom k examples
            ROW_NUMBER() OVER (
                PARTITION BY rag_version
                ORDER BY relevant_to_question_and_context_score ASC
            ) AS row_num
        FROM relevance_df
      )
      SELECT 
          request_id,
          step_id,
          rag_version,
          relevant_to_question_and_context_score,
          output_message
      FROM RankedRelevance
      WHERE row_num <= {k}
        AND rag_version in ({models_to_compare_str})
    """
)

display(bottom_k_relevant_examples_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Are my generations faithful to the given context?

# COMMAND ----------

# DBTITLE 1,Spark SQL Faithfulness Log DataFrame
models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

faithfulness_log_df = spark.sql(
    f"""
    SELECT 
        request_log.request.request_id as request_id,
        assessment_log.step_id,
        request_log.trace.app_version_id as rag_version,
        source.type as assessment_source,
        request_log.output as output_message,
        CAST(text_assessment.ratings.faithful_to_context.bool_value AS INTEGER) as faithful_to_context_value,
        text_assessment.ratings.faithful_to_context.double_value as faithful_to_context_score
    FROM (
      SELECT *
      FROM {assessment_log_table_name}
      WHERE 
        text_assessment.ratings.faithful_to_context.bool_value IS NOT NULL
        OR text_assessment.ratings.faithful_to_context.double_value IS NOT NULL
    ) assessment_log 
    JOIN {request_log_table_name} request_log 
    ON request_log.request.request_id = assessment_log.request_id
    WHERE request_log.trace.app_version_id IN ({models_to_compare_str})
"""
)

if faithfulness_log_df.count():
    # Plot the faithfulness metrics across RAG versions
    faithfulness_fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Percentage of Faithful Generations (to Context)",
            "Box Plot of Faithfulness Scores (to Context)",
        ),
    )
    faithfulness_log_df = faithfulness_log_df.withColumn("rag_version_number", get_rag_version_number(F.col("rag_version")))
    faithfulness_log_pdf = faithfulness_log_df.toPandas()

    # Bar chart of average faithfulness to context
    avg_faithfulness_pdf = faithfulness_log_pdf.groupby(["rag_version_number"]).mean().reset_index()
    faithfulness_bar = go.Bar(
        x=avg_faithfulness_pdf["faithful_to_context_value"],
        y=avg_faithfulness_pdf["rag_version_number"],
        orientation="h",
        name="Percentage of Faithful Generations",
        marker=dict(color=avg_faithfulness_pdf["rag_version_number"].apply(get_color_scale).unique()),
    )
    faithfulness_fig.add_trace(faithfulness_bar, row=1, col=1)
    faithfulness_fig.update_traces(marker_line_color = get_color_scale(0), marker_line_width = 2)

    # Box plot of faithfulness to context
    for rag_version_number in list(avg_faithfulness_pdf["rag_version_number"].unique()):
        faithfulness_box = go.Box(
            x=faithfulness_log_pdf[faithfulness_log_pdf["rag_version_number"] == rag_version_number]["faithful_to_context_score"],
            y=faithfulness_log_pdf[faithfulness_log_pdf["rag_version_number"] == rag_version_number]["rag_version_number"],
            orientation="h",
            name="Faithfulness Score (to Context)",
            marker=dict(color=get_color_scale(rag_version_number)),
        )
        faithfulness_fig.append_trace(faithfulness_box, row=1, col=2)

    # Update layout
    faithfulness_fig.update_layout(
        title={
            "text": "Faithfulness Metrics Across RAG Versions",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis=dict(
            title="Percentage Faithful to Context", side="bottom", domain=[0, 0.48]
        ),
        xaxis2=dict(
            title="Faithfulness Score (to Context)", side="bottom", domain=[0.52, 1]
        ),
        yaxis=dict(title="RAG Version"),
        showlegend=False,
    )

    # Show the plot
    faithfulness_fig.show()
else:
    print("No faithfulness results to display.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### What are the least faithful generations?

# COMMAND ----------

# DBTITLE 1,Python SQL Bottom K Faithful Examples
models_to_compare = dbutils.widgets.get("versions").split(",")
models_to_compare_str = get_sql_string_list(
    rag_versions if "*" in models_to_compare else models_to_compare)

faithfulness_log_df.createOrReplaceTempView("faithfulness_df")
k = 5
bottom_k_faithful_examples_df = spark.sql(
    f"""
      WITH RankedFaithfulness AS (
        SELECT 
            *,
            -- Rank the faithfulness scores to get the bottom k examples
            ROW_NUMBER() OVER (
                PARTITION BY rag_version
                ORDER BY faithful_to_context_score ASC
            ) AS row_num
        FROM faithfulness_df
      )
      SELECT 
          request_id,
          step_id,
          rag_version,
          faithful_to_context_score,
          output_message
      FROM RankedFaithfulness
      WHERE row_num <= {k}
        AND rag_version in ({models_to_compare_str})
    """
)

display(bottom_k_faithful_examples_df)
