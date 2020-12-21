import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

from modules import top2vec
from modules import topics_over_time as top_over_time


@st.cache(allow_output_mutation=True, show_spinner=True)
def load_model(
    paragraphs,
    from_disk=False,
    model_dir="output/distilBert_REIT-Industrial.pkl",
    path_doc_embed="output/distBert_embedding.npy"
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
            path_doc_embed="output/distBert_embedding.npy"
        )
    return model


def make_figure(df):
    """
    make figure for topic loading for words
    """
    fig = px.bar(
        df.iloc[0:10, :], x='Topic', y='Cosine similiarity',
        text="Top words", title='10 highest topic loadings')
    fig.update_layout(xaxis=dict(type='category'))
    return fig


def main():

    st.sidebar.title("Model configurations")

    df = pd.read_csv("data/CRS_processed_PyMuPDF.txt", sep='\t')
    paragraphs = df.paragraph.values.tolist()
    # model_dir = "output/distilBert_REIT-Industrial.pkl"
    model = load_model(
        paragraphs=paragraphs,
        from_disk=False,
        path_doc_embed="output/distBert_embedding.npy"
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

        if (lower_ngrams != 1 or upper_ngrams != 3 or
           min_df != 0.05 or max_df != 0.15):
            model._update_steps(documents=paragraphs, step=1)
        elif n_components != 5 or n_neighbors != 15:
            model._update_steps(documents=paragraphs, step=2)
        elif min_cluster_size != 15:
            model._update_steps(documents=paragraphs, step=3)

    topic_words, _, _ = model.get_topics()
    topic_sizes,  _ = model.get_topic_sizes()
    topics_top2Vec = pd.DataFrame(topic_words).iloc[:, 0:10]
    topics_top2Vec["size"] = topic_sizes
    expander_topics = st.beta_expander("Show topics")
    expander_topics.dataframe(topics_top2Vec)

    expander_keyword = st.beta_expander("Keyword loadings")
    keywords = expander_keyword.text_area(
        label="Input keywords (no comma required)",
        value="volunteering and community")
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


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
