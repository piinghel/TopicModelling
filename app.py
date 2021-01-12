import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

from modules import topic_identify
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(
        paragraphs,
        doc_embed
        ):
    """
    load  model
    """
    model = topic_identify.TopicIdentify(
        documents=paragraphs,
        doc_embedding=doc_embed,
        embedding_model='distiluse-base-multilingual-cased',
    )
    model.perform_steps()
    return model


@st.cache(allow_output_mutation=True, show_spinner=False)
def make_figure(df):
    """
    make figure for topic loading for words
    """
    fig = px.bar(
        df.iloc[0:10, :], x='Topic', y='Cosine similiarity',
        text="Top words", title='10 highest topic loadings')
    fig.update_layout(xaxis=dict(type='category'))
    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def count_topics(df, model, var, value, topics_words, nr_words):
    """
    counts topics
    """

    df["Topic"] = model.clusterer.labels_ + 1
    df_group = pd.DataFrame(df.groupby([var, "Topic"]).count().iloc[:, 0])
    df_group = df_group.rename(columns={df_group.columns[0]: "Count"})
    df_group_sort = (df_group.iloc[df_group.index.
                     get_level_values(0) == value, :].
                     sort_values("Count", ascending=False))
    df_group_sort["Topic"] = df_group_sort.index.get_level_values(1)
    df_group_sort["Top words"] = df_group_sort["Topic"].apply(
        (lambda x: list(topics_words.sort_values(by="topic nr").
         iloc[x, 2: 2+nr_words])))
    fig = px.bar(df_group_sort.head(10),
                 x='Topic', y="Count",
                 text="Top words", title='10 highest topic counts')
    fig.update_layout(
        xaxis=dict(type='category'),
        xaxis_title="Topic number",
        yaxis_title="Count")

    return fig


@st.cache(allow_output_mutation=True, show_spinner=False)
def construct_df_topic_words_scores(topic_words, word_scores, digits=2):
    """
    construct topics words with scores
    """
    topics_scores_df = {}
    for i, array in enumerate(word_scores.tolist()):
        store_l = []
        for j, el in enumerate(array):
            store_l.append((topic_words[i, j],
                            (float(str(word_scores[i, j])[0:5]))))
        topics_scores_df[i] = store_l
    return pd.DataFrame(topics_scores_df).T


def main():

    st.sidebar.title("Model configurations")
    st.title("Topic discovering")

    dataset = st.sidebar.selectbox(
     "Choose dataset",
     ("REIT-Industrial", "Newsgroup20 Subset"))

    if dataset == "REIT-Industrial":
        dir_doc_embed = "output/distBert_embedding_REIT-Industrial.npy"
        dir_df = "data/CRS_processed_PyMuPDF_REIT-Industrial.txt"
        example_text = "bénévolat or charity of \
liefdadigheidsdoel oder Wohltätigkeitsarbeit"

    elif dataset == "Newsgroup20 Subset":
        dir_doc_embed = "output/distBert_embedding_newsgroup_subset.npy"
        dir_df = "data/newsgroup_subset.txt"
        example_text = "Religion and god and jesus"
        st.sidebar.markdown("For more information about the newsgroup20 dataset, \
see [here](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).")

    df = pd.read_csv(dir_df, sep='\t')
    paragraphs = df.paragraph.values.tolist()
    doc_embed = np.load(dir_doc_embed)

    model = load_model(
            paragraphs=paragraphs,
            doc_embed=doc_embed
        )

    st.sidebar.write("The paragraphs and word embeddings \
were obtained using 'distiluse-base-multilingual-cased' from the \
sentence transfromer library. For more information \
see [here](https://www.sbert.net/).")
    word_embed_p = st.sidebar.beta_expander(
                    "Word embedding tuning parameters"
                    )
    word_embed_p.write(
    "For more information on the preprocessing steps for the word embeddings\
see [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).")
    add_stops_words = word_embed_p.text_area(
        "Input stopwords (separate by comma)")
    lower_ngrams = word_embed_p.number_input(
            "Lower bound ngrams",
            value=model.ngram_range[0],
            min_value=1,
            max_value=5
            )
    upper_ngrams = word_embed_p.number_input(
            "Upper bound ngrams",
            value=model.ngram_range[1],
            min_value=1,
            max_value=5
    )
    min_df = word_embed_p.slider(
            "Minimum document frequency",
            value=model.min_df,
            min_value=0.0,
            max_value=0.2,
            step=0.005
            )
    max_df = word_embed_p.slider(
            "Maximum document frequency",
            value=model.max_df,
            min_value=0.05,
            max_value=1.0,
            step=0.005
            )
    # lemmatize = st.sidebar.checkbox("Lemmatize", value=False)
    dim_reduc_p = st.sidebar.beta_expander(
                    "Dimensionality reduction tuning parameters"
                    )
    dim_reduc_p.write(
        "For more information on the dimensionality reduction algorithm \
see [here](https://umap-learn.readthedocs.io/en/latest/basic_usage.html).")
    n_components = dim_reduc_p.number_input(
            "Number of dimensions",
            value=model.n_components,
            min_value=1,
            max_value=500
            )
    n_neighbors = dim_reduc_p.number_input(
            "Number of neightsbors",
            value=model.n_neighbors,
            min_value=5,
            max_value=500
            )
    clustering_p = st.sidebar.beta_expander(
                    "Clustering tuning parameters"
                    )
    clustering_p.write(
        "For more information on the clustering algorithm \
see [here](https://hdbscan.readthedocs.io/en/latest/api.html).")
    min_cluster_size = clustering_p.number_input(
            "Minimum cluster size",
            value=model.min_cluster_size,
            min_value=5,
            max_value=500
            )

    soft_clustering = clustering_p.checkbox("Soft clustering", value=False)
    st.sidebar.write("The updating should take no longer than 3 minutes.")
    if st.sidebar.button("Update model configurations"):

        update_step = 3

        if dataset != model.dataset_name:
            model.dataset_name = dataset
            model.documents = paragraphs
            model.doc_embedding = doc_embed
            update_step = min(1, update_step)

        if model.ngram_range != (lower_ngrams, upper_ngrams):
            model.ngram_range = (lower_ngrams, upper_ngrams)
            update_step = min(1, update_step)

        if model.min_df != min_df:
            model.min_df = min_df
            update_step = min(1, update_step)

        if model.max_df != max_df:
            model.max_df = max_df
            update_step = min(1, update_step)

        if model.add_stops_words != add_stops_words:
            model.add_stops_words = add_stops_words
            update_step = min(1, update_step)

        # if model.lemmatize != lemmatize:
        #     model.lemmatize = lemmatize
        #     update_step = min(1, update_step)

        if model.n_neighbors != n_neighbors:
            model.n_neighbors = n_neighbors
            update_step = min(2, update_step)

        if model.n_components != n_components:
            model.n_components = n_components
            update_step = min(2, update_step)

        if model.min_cluster_size != min_cluster_size:
            model.min_cluster_size = min_cluster_size
            update_step = min(2, update_step)

        if model.soft_clustering != soft_clustering:
            model.soft_clustering = soft_clustering
            update_step = min(3, update_step)

        model.update(step=update_step)

    topic_words = model.topic_words
    word_scores = model.topic_word_scores
    topic_sizes = model.topic_sizes.values.tolist()
    df_topics = construct_df_topic_words_scores(
                        topic_words=topic_words,
                        word_scores=word_scores,
                        digits=2
                        ).iloc[:, 0:10]
    df_topics["size"] = topic_sizes
    df_topics["topic nr"] = list(range(0, len(topic_sizes)))
    cols = ["topic nr", "size"] + df_topics.columns.tolist()[0:9]
    df_topics = df_topics[cols].sort_values(by="size", ascending=False)

    expander_topics = st.beta_expander("Show topics")
    expander_topics.markdown("**Note:** topic 0 represents the noise topic!")
    expander_topics.dataframe(df_topics)
    expander_topics.write("\n")
    with expander_topics.beta_container():
        c1_doc, c2_doc = st.beta_columns((1, 1))
        top_nr1 = c1_doc.number_input(
            "Choose topic ",
            value=0,
            min_value=0,
            max_value=(len(model.topic_sizes)-1)
            )
        st.write(len(model.topic_sizes))
        top_nr2 = c2_doc.number_input(
            "Choose topic",
            value=1,
            min_value=0,
            max_value=(len(model.topic_sizes)-1)
        )
        c1_doc.pyplot(model.generate_topic_wordcloud(topic_num=top_nr1))
        c2_doc.pyplot(model.generate_topic_wordcloud(topic_num=top_nr2))

    expander_keyword_topics = st.beta_expander(
        "Show keyword/sentence topic loadings")
    keywords_top = expander_keyword_topics.text_area(
        label="Input keywords for topic search (no comma required) or \
small paragraphs (max 125 words).",
        value=example_text)
    keyword_embed = model.embedder.encode([keywords_top])
    res = cosine_similarity(keyword_embed, model.topic_vectors)
    scores = round(pd.DataFrame(res, index=["Cosine similiarity"]).T, 3)
    scores["Topic"] = list(range(0, len(scores)))
    scores["Top words"] = scores["Topic"].apply(
        lambda x: list(df_topics.sort_values(by="topic nr").iloc[x, 2:5]))
    scores.sort_values(by="Cosine similiarity", ascending=False, inplace=True)
    fig = make_figure(scores)
    expander_keyword_topics.plotly_chart(fig, use_container_width=True)

    expander_sim_matrix = st.beta_expander(
        "Topic similarity")
    fig = model.topic_similiarity(reduced=False)
    expander_sim_matrix.plotly_chart(fig, use_container_width=True)

    expander_documents = st.beta_expander(
        "Search topic by documents")
    with expander_documents.beta_container():
        c1_doc, c2_doc = st.beta_columns((1, 1))
        topic_num = (
            c1_doc.number_input(
                "Choose topic number",
                value=0,
                min_value=0,
                max_value=(len(model.topic_sizes)-1)
                )
            )
        num_docs = c2_doc.number_input(
            "Number of documents to show",
            value=3,
            min_value=0,
            max_value=10
            )

    idx, scores = (
                model.search_topic_by_documents(
                    topic_nr=topic_num, n=num_docs))
    for score, id in zip(scores, idx):
        expander_documents.write(f"Document: {id}, Score: {str(score)[0:5]}")
        expander_documents.write(df.iloc[id, :].paragraph)
        expander_documents.write()

    expander_keywords_docs = st.beta_expander(
        "Search documents by keywords")
    with expander_keywords_docs.beta_container():
        c1_key, c2_key = st.beta_columns((1, 1))
        keywords_doc = c1_key.text_area(
            label="Input keywords for documents search (no comma required) or \
small paragraphs (max 125 words).",
            value=example_text)
        num_docs = c2_key.number_input(
            "Choose number of documents to show",
            value=3,
            min_value=0,
            max_value=10
        )

    idx, scores = (
                model.documents_by_keywords(
                    keywords=keywords_doc, n=num_docs))
    for score, id in zip(scores, idx):
        expander_keywords_docs.write(f"Document: {id}, \
Score: {str(score)[0:5]}")
        expander_keywords_docs.write(df.iloc[id, :].paragraph)
        expander_keywords_docs.write()

    if dataset == "REIT-Industrial":
        expander_count_topics = st.beta_expander(
                "Count topics for a chosen variable and value")
        with expander_count_topics.beta_container():
            c1_count, c2_count = st.beta_columns((1, 1))
            var = c1_count.selectbox(
                "Choose variable",
                ("company", "industry", "sector", "filename"))
            values = df[var].unique()
            value = c2_count.selectbox("Choose value", values)
            fig_count_topics = count_topics(
                df=df,
                model=model,
                var=var,
                value=value,
                topics_words=df_topics,
                nr_words=3
            )

            expander_count_topics.plotly_chart(
                fig_count_topics, use_container_width=True
            )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
