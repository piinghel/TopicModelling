import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

from modules import top2vec
from modules import topics_over_time as top_over_time
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(
    paragraphs,
    from_disk=False,
    model_dir="output/distilBert_REIT-Industrial.pkl",
    path_doc_embed=None
):
    """
    load top2vec model
    """

    if from_disk:
        with open(model_dir, 'rb') as file:
            model = pickle.load(file)

    else:
        model = top2vec.Top2Vec(
            documents=paragraphs,
            embedding_model='distiluse-base-multilingual-cased',
            load_doc_embed=True,
            save_doc_embed=False,
            path_doc_embed=path_doc_embed
        )
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

    df["Topic"] = model.clustering.labels_
    df_group = pd.DataFrame(df.groupby([var, "Topic"]).count().iloc[:, 0])
    df_group = df_group.rename(columns={df_group.columns[0]: "Count"})
    df_group_sort = (df_group.iloc[df_group.index.
                     get_level_values(0) == value, :].
                     sort_values("Count", ascending=False))
    df_group_sort["Topic"] = df_group_sort.index.get_level_values(1)
    df_group_sort["Top words"] = df_group_sort["Topic"].apply(
        (lambda x: list(topics_words.iloc[x, 0:nr_words]) if
         x >= 0 else list(["None"])))

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
                            round(word_scores[i, j], digits)))
        topics_scores_df[i] = store_l
    return pd.DataFrame(topics_scores_df).T


def main():

    st.sidebar.title("Model configurations")

    # dataset = st.sidebar.selectbox(
    #  "Choose dataset",
    #  ("REIT-Industrial", "All documents"))

    # if dataset == "REIT-Industrial":
    #     dir_doc_embed = "output/distBert_embedding_REIT-Industrial.npy"
    #     dir_df = "data/CRS_processed_PyMuPDF_REIT-Industrial.txt"

    # elif dataset == "All documents":
    #     dir_doc_embed = "output/distBert_embedding_all-doc.npy"
    #     dir_df = "data/CRS_processed_PyMuPDF_All-Doc.txt"

    dir_doc_embed = "output/distBert_embedding_REIT-Industrial.npy"
    dir_df = "data/CRS_processed_PyMuPDF_REIT-Industrial.txt"

    df = pd.read_csv(dir_df, sep='\t')
    paragraphs = df.paragraph.values.tolist()
    # model_dir = "output/distilBert_REIT-Industrial.pkl"
    model = load_model(
        paragraphs=paragraphs,
        from_disk=False,
        path_doc_embed=dir_doc_embed
    )

    add_stops_words = st.sidebar.text_area(
        "Input stopwords (separate by comma)")
    lower_ngrams = st.sidebar.number_input(
            "Lower bound ngrams",
            value=1,
            min_value=1,
            max_value=5
            )
    upper_ngrams = st.sidebar.number_input(
            "Upper bound ngrams",
            value=3,
            min_value=1,
            max_value=5
    )
    min_df = st.sidebar.slider(
            "Minimum document frequency",
            value=0.005,
            min_value=0.0,
            max_value=0.2,
            step=0.005
            )
    max_df = st.sidebar.slider(
            "Maximum document frequency",
            value=0.15,
            min_value=0.05,
            max_value=1.0,
            step=0.005
            )
    # lemmatize = st.sidebar.checkbox("Lemmatize", value=False)
    lemmatize = False
    n_components = st.sidebar.number_input(
            "Number of dimension (dimensionality reduction)",
            value=5,
            min_value=1,
            max_value=500
            )
    n_neighbors = st.sidebar.number_input(
            "Number of neightsbors (dimensionality reduction)",
            value=15,
            min_value=5,
            max_value=500
            )
    min_cluster_size = st.sidebar.number_input(
            "Minimum cluster size (clustering)",
            value=15,
            min_value=5,
            max_value=500
            )

    if st.sidebar.button("Update model configurations"):

        model.ngram_range = (lower_ngrams, upper_ngrams)
        model.min_df = min_df
        model.max_df = max_df
        model.n_neighbors = n_neighbors
        model.n_components = n_components
        model.min_cluster_size = min_cluster_size
        model.add_stops_words = add_stops_words
        model.random_seed = 69
        model.lemmatize = lemmatize

        if (lower_ngrams != 1 or upper_ngrams != 3 or
           min_df != 0.05 or max_df != 0.15 or lemmatize):
            model._update_steps(documents=paragraphs, step=1)
        elif n_components != 5 or n_neighbors != 15:
            model._update_steps(documents=paragraphs, step=2)
        elif min_cluster_size != 15:
            model._update_steps(documents=paragraphs, step=3)

    topic_words, word_scores, _ = model.get_topics()
    topic_sizes,  _ = model.get_topic_sizes()
    topics_top2Vec = construct_df_topic_words_scores(
                        topic_words=topic_words,
                        word_scores=word_scores,
                        digits=2
                        ).iloc[:, 0:10]
    topics_top2Vec["size"] = topic_sizes
    expander_topics = st.beta_expander("Show topics")
    expander_topics.dataframe(topics_top2Vec)
    expander_topics.write("\n")
    with expander_topics.beta_container():
        c1_doc, c2_doc = st.beta_columns((1, 1))
        top_nr1 = c1_doc.number_input(
            "Choose topic ",
            value=0,
            min_value=0,
            max_value=model.get_num_topics())
        top_nr2 = c2_doc.number_input(
            "Choose topic",
            value=1,
            min_value=0,
            max_value=model.get_num_topics())
        c1_doc.pyplot(model.generate_topic_wordcloud(topic_num=top_nr1))
        c2_doc.pyplot(model.generate_topic_wordcloud(topic_num=top_nr2))

    expander_keyword = st.beta_expander("Show keyword/sentence topic loadings")
    keywords = expander_keyword.text_area(
        label="Input keywords (no comma required) or \
small paragraphs (max 125 words).",
        value="bénévolat or charity of \
liefdadigheidsdoel oder Wohltätigkeitsarbeit")
    keyword_embed = model.embed([keywords])
    res = cosine_similarity(keyword_embed, model.topic_vectors)
    scores = pd.DataFrame(res, index=["Cosine similiarity"]).T
    scores["Topic"] = list(range(0, len(scores)))
    scores["Top words"] = scores["Topic"].apply(
        lambda x: list(topics_top2Vec.iloc[x, 0:3]))
    scores.sort_values(by="Cosine similiarity", ascending=False, inplace=True)
    fig = make_figure(scores)
    expander_keyword.plotly_chart(fig, use_container_width=True)

    expander_documents = st.beta_expander(
        "Show most similar documents for topic")
    with expander_documents.beta_container():
        c1_doc, c2_doc = st.beta_columns((1, 1))
        topic_num = c1_doc.number_input("Choose topic number", value=1)
        num_docs = c2_doc.number_input(
            "Choose number of documents to show", value=3)

    documents, document_scores, document_ids = (
                model.search_documents_by_topic(
                    topic_num=topic_num, num_docs=num_docs))
    for doc, score, doc_id in zip(documents, document_scores, document_ids):
        expander_documents.write(f"Document: {doc_id}, Filename (Company and year):  \
{df.iloc[doc_id,:].filename}, Score: {round(score,2)}")
        expander_documents.write(doc)
        expander_documents.write()

    expander_topics_time = st.beta_expander("Show topics over time")
    fig_topics_year, _, _ = top_over_time.topics_over_time(
            model=model,
            df_paragraphs=df,
            num_topics=5,
            reduced=True,
            make_fig=True
    )
    expander_topics_time.plotly_chart(
        fig_topics_year, use_container_width=True)

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
            topics_words=topics_top2Vec,
            nr_words=3
        )

        expander_count_topics.plotly_chart(
            fig_count_topics, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
