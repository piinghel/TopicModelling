import pandas as pd
import plotly.express as px


def topics_over_time(
    model,
    df_paragraphs,
    num_topics=10,
    reduced=True,
    num_words=1,
    make_fig=True
):
    """
    computes importance of topics over time, optionality to visualize
    """

    if reduced:
        _ = model.hierarchical_topic_reduction(num_topics=num_topics)
    topic_words, _, _ = model.get_topics(reduced=True)
    topic_sizes,  topic_nums = model.get_topic_sizes(reduced=True)
    topics_df = pd.DataFrame(topic_words).iloc[:, 0:10]
    topics_df["reduced"] = model.get_topic_hierarchy()
    topics_df["size"] = topic_sizes
    # number of paragraphs by year
    total_paragraphs_year = df_paragraphs.groupby("year").count()["company"]

    # compute percentage over time
    topics_over_time_perc = {}
    for i, size in enumerate(topic_sizes):
        _, _, idx = model.search_documents_by_topic(
                        topic_num=i, num_docs=size, reduced=reduced)
        out_perc = (df_paragraphs.iloc[idx, :].
                    groupby("year").count()["company"] / total_paragraphs_year)
        topics_over_time_perc[i] = out_perc
    df_wide_year = pd.DataFrame(topics_over_time_perc)
    df_wide_year["year"] = df_wide_year.index

    if not make_fig:
        return _, topics_df, df_wide_year
    else:
        # to long format
        df_long = pd.melt(df_wide_year, id_vars="year",
                          value_vars=df_wide_year.columns[0:-1])
        # map topic numbers to first three words
        map_dic = {i: tuple(row[0:num_words]) for i,
                   row in topics_df.iterrows()}
        df_long["Topic"] = df_long["variable"].apply(lambda x: map_dic.get(x))
        # visualize
        fig = px.area(df_long, x="year", y="value", color="Topic")
        fig.update_xaxes(title_text='Year')
        fig.update_yaxes(title_text='Topic percentage (stacked)')

        return fig, topics_df, df_wide_year
