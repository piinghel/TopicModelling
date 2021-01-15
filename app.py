import streamlit as st
from modules import helper_module_app as helper
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():

    """
    glues all parts together
    """

    st.sidebar.title("Model configurations")
    st.title("Topic discovering")
    # choose dataset
    dataset = st.sidebar.selectbox(
        "Choose dataset",
        (["REIT-Industrial"])
     )
    # loads data and embeddings
    sentence_model = helper.load_sentence_model(
        model_name="distiluse-base-multilingual-cased"
    )
    df, doc_embed, example_text = helper.load_data(dataset)
    paragraphs = df.paragraph.values.tolist()
    if dataset == "Newsgroup20 Subset":
        st.sidebar.markdown("For more information about the newsgroup20 dataset, \
see [here](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).")


    # loads model
    model = helper.load_model(
            paragraphs=paragraphs,
            sentence_model=sentence_model,
            doc_embedding=doc_embed
        )
    # add company names as stop words
    if dataset == "REIT-Industrial":
        model.add_stops_words = list(df.company.unique())
    model.dataset_name = dataset


    st.sidebar.markdown("The paragraphs and word embeddings were obtained using \
distiluse-base-multilingual-cased from the sentence transfromer library. \
For more information see [here](https://www.sbert.net/).")

    # parameters word embeddings
    model, stop_words, lower_ngrams, upper_ngrams, min_df, max_df = (
        helper.params_word_embed(model)
    )
    # parameters dim reduction
    model, n_components, n_neighbors, densmap = (
        helper.params_dim_red(model)
    )
    # parameters clustering
    (model, min_cluster_size,
     min_samples, selection_epsilon) = (
        helper.params_clustering(model)
    )
    st.sidebar.markdown("Hit the **initialiation checkbox**, \
once the model has been intialialized. This will skip the initailization \
part each time you change parameter values.")
    skip_initial = st.sidebar.checkbox("Skip initializating", value=False)
    if not skip_initial:
        with st.spinner("Initializing model. \
Hit the intialization checkbox once the model has been intitialized. \
This will save time."):
            model.perform_steps()
    st.sidebar.markdown("Do not forget to hit the button **update model configurations** \
when changing the parameter values. \
The updating should take no longer than 3 minutes.")

    if st.sidebar.button("Update model configurations"):
        model = helper.update_model_steps(
                    model=model,
                    dataset=dataset,
                    doc_embed=doc_embed,
                    paragraphs=paragraphs,
                    lower_ngrams=lower_ngrams,
                    upper_ngrams=upper_ngrams,
                    min_df=min_df,
                    max_df=max_df,
                    stop_words=stop_words,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    densmap=densmap,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    selection_epsilon=selection_epsilon
                )

    # apply topic reduction?
    topic_reduction = st.sidebar.checkbox("Topic reduction", value=False)
    if topic_reduction:
        st.sidebar.markdown("Do not foret to hit the **update number of topics** \
button when changing the number of topics.")

    # Section 1: Table with topics
    expander_topics, reduced_topic_sec_tw,  = (
        helper.display_topics(model, topic_reduction)
    )
    expander_topics.write("\n")
    helper.display_word_cloud(
        model, expander_topics, reduced_topic_sec_tw
    )

    # Section 2: Keyword loadings on topics
    helper.topic_keywords(
        model, example_text, topic_reduction
    )

    # Section 3: Topic similarity matrix
    helper.show_similarity_matrix(model, topic_reduction)

    # Section 4: Search most relevant documents for a topic cluster
    helper.most_relevant_doc_top(df, model, topic_reduction)

    # Section 5: Search documents by keywords
    helper.documents_keywords(model, df, example_text)

    # Section 6: Search documents by keywords
    if dataset == "REIT-Industrial":
        helper.topic_size_vars_value(model, df, topic_reduction)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
