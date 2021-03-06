import graphviz as graphviz
import streamlit as st
from modules import helper_module_app as helper
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():

    """
    glues all parts together
    """

    st.sidebar.title("Model configurations")
    st.title("Topic discovering")
    st.write("")

    graph = graphviz.Digraph()
    graph.edge('Input model embeddings from documents',
               'Dimension reduction (Step 2)')
    graph.edge('Input documents', 'Extract keywords \
and perform word embeddings (Step 1)')
    graph.edge('Dimension reduction (Step 2)', 'Clustering (Step 3)')
    graph.edge('Clustering (Step 3)', 'Construct topic vectors')
    graph.edge('Construct topic vectors', 'Attach keywords \
(ngrams) to each topic vector')
    graph.edge('Extract keywords and perform word embeddings (Step 1)', 'Attach keywords \
(ngrams) to each topic vector')
    st.graphviz_chart(graph)

    # choose dataset
    dataset = st.sidebar.selectbox(
        "Choose dataset",
        (["REIT-Industrial"])
     )

    df, doc_embed, example_text = helper.load_data(dataset)
    original_data_expander = st.beta_expander("Show raw data (source)")

    paragraphs = df.paragraph.values.tolist()
    if dataset == "Newsgroup20 Subset":
        st.sidebar.markdown("For more information about the newsgroup20 dataset, \
see [here](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).")

    # loads model

    if dataset == "REIT-Industrial":
        add_stop_words = ["Alexandrias", "Alexandria", "Yellow", "Yellows"]
        add_stop_words = list(df.company.unique()) + add_stop_words
        model = helper.load_model(add_stops_words=add_stop_words)
    else:
        model = helper.load_model()

    if model.dataset_name != dataset:
        with st.spinner("Change of dataset: updating step 1 to 3"):
            # update and reset arguments
            model.doc_embedding = doc_embed
            model.documents = paragraphs
            model.dataset_name = dataset
            model.topic_sizes_reduced = None
            model.topic_vectors_reduced = None
            model.topic_words_reduced = None
            model.topic_word_scores_reduced = None
            model.topic_hierarchy = None
            model.perform_steps()

    st.sidebar.markdown("The paragraph and word embeddings were obtained using \
[distiluse-base-multilingual-cased](https://arxiv.org/abs/1910.01108) \
from the [sentence transformer library](https://www.sbert.net/).")

    with original_data_expander.beta_container():
        original_data_expander.markdown("Here you see the (sources) of the original \
data.")
        if dataset == "REIT-Industrial":
            show_p = original_data_expander.checkbox(
                "Show extracted paragraphs", value=False
            )
            if show_p:
                original_data_expander.dataframe(df)
            else:
                original_data_expander.dataframe(
                    df.iloc[:, 0:5].drop_duplicates().
                    reset_index(inplace=False, drop=True)
                )
        else:
            original_data_expander.dataframe(df.iloc[:, 1])

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

    st.sidebar.markdown("Do not forget to hit the **update model configurations** \
button when changing the parameter values. \
The updating should take no longer than 1 minute.")

    if st.sidebar.button("Update model configurations"):
        model = helper.update_model_steps(
                    model=model,
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
    # expander_topics.write("\n")
    # helper.display_word_cloud(
    #     model, expander_topics, reduced_topic_sec_tw
    # )

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
