import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from modules import topic_identify
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
st.set_option('deprecation.showPyplotGlobalUse', False)


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """
    Generates a link allowing the data in a
    given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,\
{b64.decode()}" download="topics_word_scores.xlsx">Download csv file</a>'


def construct_topics_df(model, reduced=False):
    """
    construct dataframe containing topics words, scores, and sizes
    """
    if reduced:

        topic_words = model.topic_words_reduced
        word_scores = model.topic_word_scores_reduced
        topic_sizes = model.topic_sizes_reduced
    else:

        topic_words = model.topic_words
        word_scores = model.topic_word_scores
        topic_sizes = model.topic_sizes.values.tolist()

    df_topics = construct_df_topic_words_scores(
                topic_words=topic_words,
                word_scores=word_scores,
                digits=2
                ).iloc[:, 0:10]

    df_topics["size"] = topic_sizes
    if reduced:
        df_topics["hierarchy"] = model.topic_hierarchy
        cols = ["hierarchy", "size"] + df_topics.columns.tolist()[0:10]
    else:
        df_topics["topic nr"] = list(range(0, len(topic_sizes)))
        cols = ["topic nr", "size"] + df_topics.columns.tolist()[0:10]

    return df_topics[cols]


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


@st.cache(allow_output_mutation=True, show_spinner=False, max_entries=1)
def load_model():
    """
    load  model
    """
    model_name = "distiluse-base-multilingual-cased"
    df, doc_embed, example_text = load_data("REIT-Industrial")
    paragraphs = df.paragraph.values.tolist()

    with st.spinner(f'Loading model {model_name}'):
        sentence_model = SentenceTransformer(model_name)
    model = topic_identify.TopicIdentify(
            documents=paragraphs,
            embedding_model=sentence_model,
            doc_embedding=doc_embed
        )
    model.dataset_name = "REIT-Industrial"
    with st.spinner("initialization: performing \
word embedding (step 1),  dimensionality reduction (step 2), \
and clustering (step 3)"):
        model.perform_steps()
    return model


def make_figure(df, x):
    """
    make figure for topic loading for words
    """
    n = min(df.shape[0], 10)
    fig = px.bar(
        df.iloc[0:n, :], x=x, y='Cosine similiarity',
        text="Top words", title=f'{n} highest topic loadings')
    fig.update_layout(xaxis=dict(type='category'))
    return fig


def count_topics(df, model, var, value, nr_words, reduced):
    """
    counts topics
    """
    topics_words = construct_topics_df(model, reduced)
    if reduced:
        res = np.inner(model.doc_embedding, model.topic_vectors_reduced)
        topic_idx = np.flip(np.argsort(res, axis=1), axis=1)
        df["Topic"] = [model.topic_hierarchy[topic_idx[i, 0]]
                       for i in range(0, len(topic_idx))]

    else:
        df["Topic"] = model.clusterer.labels_ + 1
    df_group = pd.DataFrame(df.groupby([var, "Topic"]).count().iloc[:, 0])
    df_group = df_group.rename(columns={df_group.columns[0]: "Count"})
    df_group = (
        df_group.iloc[df_group.index.get_level_values(0) == value, :]
    )
    df_group["Topic"] = df_group.index.get_level_values(1)
    if reduced:
        store_idx_topic_hierarchy = []
        df_hier = pd.DataFrame(model.topic_hierarchy)
        for idx, row in df_group.iterrows():
            idx_th = df_hier[df_hier.iloc[:, 0] == row["Topic"]].index[0]
            store_idx_topic_hierarchy.append(idx_th)
        df_group["Topic idx"] = store_idx_topic_hierarchy
        df_group["Top words"] = df_group["Topic idx"].apply(
            (lambda x: list(topics_words.iloc[x, 2: 2+nr_words])))
    else:
        df_group["Top words"] = df_group["Topic"].apply(
            (lambda x: list(topics_words.iloc[x, 2: 2+nr_words])))

    n = min(df_group.shape[0], 10)
    fig = px.bar(
        df_group.sort_values(by="Count", ascending=False).head(n),
        x='Topic',
        y="Count",
        text="Top words",
        title=f'{n} highest topic counts'
    )
    fig.update_layout(
        xaxis=dict(type='category'),
        xaxis_title="Topic number",
        yaxis_title="Count"
    )

    return fig


def load_data(dataset):
    """
    loads paragraphs and document embeddings
    from the chosen dataset
    """
    if dataset == "REIT-Industrial":
        dir_doc_embed = "data/distBert_embedding_REIT-Industrial.npy"
        dir_df = "data/CRS_processed_PyMuPDF_REIT-Industrial.txt"
        example_text = "bénévolat or charity of \
liefdadigheidsdoel oder Wohltätigkeitsarbeit"

    elif dataset == "Newsgroup20 Subset":
        dir_doc_embed = "data/distBert_embedding_newsgroup_subset.npy"
        dir_df = "data/newsgroup_subset.txt"
        example_text = "Religion and god and jesus"

    df = pd.read_csv(dir_df, sep='\t')
    doc_embed = np.load(dir_doc_embed)
    return df, doc_embed, example_text


def params_word_embed(model):
    """
    parameters word embeddings
    """
    word_embed_p = st.sidebar.beta_expander(
            "Step 1: Word embedding tuning parameters"
            )

    word_embed_p.write(
        "For more information on the preprocessing steps for the word embeddings \
see [here](https://scikit-learn.org/stable/modules/generated/\
sklearn.feature_extraction.text.CountVectorizer.html).")
    stop_words = word_embed_p.text_area(
                "Input stopwords (separate by comma)",
                value=','.join([str(elem) for elem in model.add_stops_words])
            )
    lower_ngrams = word_embed_p.slider(
                "Lower bound ngrams",
                value=model.ngram_range[0],
                min_value=1,
                max_value=5
                )
    upper_ngrams = word_embed_p.slider(
                "Upper bound ngrams",
                value=model.ngram_range[1],
                min_value=1,
                max_value=5
        )
    min_df = word_embed_p.slider(
                "Minimum document frequency (%)",
                value=model.min_df,
                min_value=0.0,
                max_value=0.2,
                step=0.005
                )
    max_df = word_embed_p.slider(
                "Maximum document frequency (%)",
                value=model.max_df,
                min_value=0.05,
                max_value=1.0,
                step=0.005
                )
    #lemmatize = st.sidebar.checkbox("Lemmatize", value=False)

    return model, stop_words, lower_ngrams, upper_ngrams, min_df, max_df


def params_dim_red(model):
    """
    """
    dim_reduc_p = st.sidebar.beta_expander(
            "Step 2: Dimensionality reduction tuning parameters"
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

    densmap = dim_reduc_p.checkbox(
            "apply densmap \
(preserve more information about the relative local density of data)",
            value=model.densmap
    )

    return model, n_components, n_neighbors, densmap


def params_clustering(model):

    clustering = st.sidebar.beta_expander(
            "Step 3: Clustering tuning parameters"
                    )
    clustering.write(
        "For more information on the clustering algorithm \
see [here](https://hdbscan.readthedocs.io/en/latest/api.html).")

    min_cluster_size = clustering.number_input(
            "Minimum cluster size",
            value=model.min_cluster_size,
            min_value=5,
            max_value=100
            )
    min_samples = clustering.number_input(
            "The number of samples in a neighbourhood for a \
point to be considered a core point",
            value=model.min_samples,
            min_value=5,
            max_value=100
            )
    selection_epsilon = clustering.slider(
            "A distance threshold. Clusters below this value will be merged",
            value=model.cluster_selection_epsilon,
            min_value=0.0,
            max_value=0.99,
            step=0.01
    )

    return (model, min_cluster_size,
            min_samples, selection_epsilon
            )


def update_model_steps(
        model,
        doc_embed,
        paragraphs,
        lower_ngrams,
        upper_ngrams,
        min_df,
        max_df,
        stop_words,
        n_neighbors,
        n_components,
        densmap,
        min_cluster_size,
        min_samples,
        selection_epsilon
        ):

    update_step = 4

    if model.ngram_range != (lower_ngrams, upper_ngrams):
        model.ngram_range = (lower_ngrams, upper_ngrams)
        update_step = min(1, update_step)

    if model.min_df != min_df:
        model.min_df = min_df
        update_step = min(1, update_step)

    if model.max_df != max_df:
        model.max_df = max_df
        update_step = min(1, update_step)

    model_stop_word_str = ','.join([str(elem) for elem in model.add_stops_words])
    if model_stop_word_str != stop_words:
        model.add_stops_words = stop_words

    # if model.lemmatize != lemmatize:
    #     model.lemmatize = lemmatize
    #     update_step = min(1, update_step)

    if model.n_neighbors != n_neighbors:
        model.n_neighbors = n_neighbors
        update_step = min(2, update_step)

    if model.n_components != n_components:
        model.n_components = n_components
        update_step = min(2, update_step)

    if model.densmap != densmap:
        model.densmap = densmap
        update_step = min(2, update_step)

    if model.min_cluster_size != min_cluster_size:
        model.min_cluster_size = min_cluster_size
        update_step = min(3, update_step)

    if model.min_samples != min_samples:
        model.min_samples = min_samples
        update_step = min(3, update_step)

    if model.cluster_selection_epsilon != selection_epsilon:
        model.cluster_selection_epsilon = selection_epsilon
        update_step = min(3, update_step)

    if update_step != 4:
        if update_step == 3:
            with st.spinner('Updating step 3'):
                model.update(step=update_step)
        else:
            with st.spinner(f'Updating step {update_step} to 3'):
                model.update(step=update_step)

    return model


def display_topics(model, topic_reduction):
    """
    displays topics
    """
    if model.topic_words is not None:
        if topic_reduction:
            nr_topics_red = st.sidebar.number_input(
                            label="Choose Number of topics",
                            min_value=2,
                            value=len(model.topic_sizes),
                            max_value=len(model.topic_sizes)
                            )
            if st.sidebar.button("Update number of topics"):
                with st.spinner("updating number op topics!"):
                    model.topic_reduction(num_topics=nr_topics_red)

    expander_topics = st.beta_expander("Show topics")
    topic_red_sec_tw = False
    if model.topic_words_reduced is not None:
        if topic_reduction:
            topic_red_sec_tw = expander_topics.checkbox(
                label="Show dataframe with reduced number of topics",
                value=False
            )
    df_topics = construct_topics_df(model, topic_red_sec_tw)
    df_topics = df_topics.sort_values(by="size", ascending=False)
    df_topics.reset_index(inplace=True, drop=True)
    expander_topics.markdown("This table shows the 10 most representive words for each topic \
along with the word scores, and topic sizes.")
    expander_topics.markdown("**Note:** topic 0 represents the noise topic!")
    expander_topics.dataframe(df_topics)
    expander_topics.markdown(
        get_table_download_link(df_topics), unsafe_allow_html=True
    )
    return expander_topics, topic_red_sec_tw


def display_word_cloud(model, expander_topics, reduced):

    with expander_topics.beta_container():
        c1_doc, c2_doc = st.beta_columns((1, 1))
        df_topics = construct_topics_df(model, reduced)
        top_nr1 = c1_doc.selectbox(
            "Choose topic (hierarchy)",
            options=df_topics.iloc[:, 0].values.tolist(),
            index=0
            )
        top1_idx = df_topics[df_topics.iloc[:, 0] == top_nr1].index[0]
        top_nr2 = c2_doc.selectbox(
            "Choose topic (hierarchy)",
            options=df_topics.iloc[:, 0].tolist(),
            index=1
        )
        top2_idx = df_topics[df_topics.iloc[:, 0] == top_nr2].index[0]
        fig1 = model.generate_topic_wordcloud(
            topic_num=top1_idx,
            title=top_nr1,
            reduced=reduced,
            background_color="white"
            )
        c1_doc.pyplot(fig1)
        fig2 = model.generate_topic_wordcloud(
            topic_num=top2_idx,
            title=top_nr2,
            reduced=reduced,
            background_color="white"
        )
        c2_doc.pyplot(fig2)


@st.cache(show_spinner=False, allow_output_mutation=True, max_entries=1)
def embed_keywords(keywords):
    """
    embeds keywords or a paragraph
    """
    model = load_model()
    return model._l2_normalize(
            model.embedding_model.encode(keywords)
    )


def topic_keywords(model, text, topic_reduction=False):
    """
    computes most similar topics for a given set of keywords
    """

    expander_keyword_topics = st.beta_expander(
        "Show keyword/sentence topic loadings"
    )

    topic_red_sec_kw = False
    if model.topic_words_reduced is not None:
        if topic_reduction:
            topic_red_sec_kw = (
                expander_keyword_topics.checkbox("On reduced topics")
            )
    keywords_input = expander_keyword_topics.text_area(
        label="Input keywords for topic search (no comma required) or \
small paragraphs (max 125 words).",
        value=text
    )
    keyword_embed = embed_keywords(keywords_input)
    keyword_embed = keyword_embed.reshape(1, len(keyword_embed))
    if topic_red_sec_kw:
        sims_vector = cosine_similarity(
            keyword_embed, model.topic_vectors_reduced
        )
        df_topics = construct_topics_df(model, True)
        scores = round(pd.DataFrame(
            sims_vector, index=["Cosine similiarity"]).T, 3
        )
        scores["Topic"] = list(range(0, len(scores)))
        scores["Top words"] = scores["Topic"].apply(
            lambda x: list(df_topics.iloc[x, 2:5]))
        scores["Hierarchy"] = [str(i)
                               for i in df_topics.hierarchy.values.tolist()]
        scores.sort_values(
            by="Cosine similiarity", ascending=False, inplace=True
        )
        fig = make_figure(scores, x="Hierarchy")
        expander_keyword_topics.plotly_chart(fig, use_container_width=True)
    else:
        sims_vector = cosine_similarity(keyword_embed, model.topic_vectors)
        df_topics = construct_topics_df(model, False)
        scores = round(pd.DataFrame(
            sims_vector, index=["Cosine similiarity"]).T, 3
        )
        scores["Topic"] = list(range(0, len(scores)))
        scores["Top words"] = scores["Topic"].apply(
            lambda x: list(df_topics.iloc[x, 2:5]))
        scores.sort_values(
            by="Cosine similiarity", ascending=False, inplace=True
        )
        fig = make_figure(scores, x="Topic")
        expander_keyword_topics.plotly_chart(fig, use_container_width=True)


def show_similarity_matrix(model, topic_reduction=False):
    """
    Computes topic similarity matrix
    """
    expander_sec_sm = st.beta_expander(
            "Topic similarity"
        )
    reduced_sec_ts = False
    if model.topic_words_reduced is not None:
        if topic_reduction:
            reduced_sec_ts = expander_sec_sm.checkbox(
                "Similiarty on reduced number of topics", value=False
            )
    fig = model.topic_similiarity(reduced=reduced_sec_ts)
    expander_sec_sm.plotly_chart(fig, use_container_width=True)


def most_relevant_doc_top(df, model, topic_reduction):
    """
    displays the most relevant documents for a chosen topic
    """
    expander_documents = st.beta_expander(
        "Search topic by documents")
    with expander_documents.beta_container():
        c1_doc, c2_doc = st.beta_columns((1, 1))
        topic_reduction_doc = False
        if model.topic_words_reduced is not None:
            if topic_reduction:
                topic_reduction_doc = expander_documents.checkbox(
                    "Apply topic reduction", value=False
                )
        topic_df = construct_topics_df(model, topic_reduction_doc)
        topic_num = (
            c1_doc.selectbox(
                "Choose topic number",
                options=topic_df.iloc[:, 0],
                index=0
            )
        )
        topix_idx = topic_df[topic_df.iloc[:, 0] == topic_num].index[0]
        num_docs = c2_doc.number_input(
            "Number of documents to show",
            value=3,
            min_value=0,
            max_value=10
            )

    idx, scores = model.search_topic_by_documents(
                topic_nr=topix_idx,
                n=num_docs,
                reduced=topic_reduction_doc
    )
    for score, id in zip(scores, idx):
        expander_documents.write(f"Document: {id}, Score: {str(score)[0:5]}")
        expander_documents.write(df.iloc[id, :].paragraph)
        expander_documents.write()


def topic_size_vars_value(model, df, topic_reduction):
    """
    displays the topic size for a given variable and value
    """
    expander_count_topics = st.beta_expander(
            "Count topics for a chosen variable and value")
    with expander_count_topics.beta_container():
        topic_red_sec_vw = False
        if model.topic_words_reduced is not None:
            if topic_reduction:
                topic_red_sec_vw = (
                    expander_count_topics.checkbox("On reduced topics?")
                )
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
            nr_words=3,
            reduced=topic_red_sec_vw
        )

        expander_count_topics.plotly_chart(
            fig_count_topics, use_container_width=True
        )


def documents_keywords(model, df, text):
    """
    searches documents containing similar info as the keywords
    """

    expander_keywords_docs = st.beta_expander(
        "Search documents by keywords")
    with expander_keywords_docs.beta_container():
        c1_key, c2_key = st.beta_columns((1, 1))

        keywords_doc = c1_key.text_area(
            label="Input keywords for documents search (no comma required) or \
small paragraphs (max 125 words).",
            value=text
        )

        num_docs = c2_key.number_input(
            "Choose number of documents to show",
            value=3,
            min_value=0,
            max_value=10
        )

    key_word_embed = embed_keywords(keywords_doc)
    res = np.inner(key_word_embed, model.doc_embedding)
    most_similar_idx = np.flip(np.argsort(res))
    idx = most_similar_idx[0:num_docs]
    scores = np.take(res, most_similar_idx)[0:num_docs]
    for score, id in zip(scores, idx):
        expander_keywords_docs.write(f"Document: {id}, \
Score: {str(score)[0:5]}")
        expander_keywords_docs.write(df.iloc[id, :].paragraph)
        expander_keywords_docs.write()
