import umap
import hdbscan
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import re
from textblob import TextBlob
from sklearn.preprocessing import normalize
from tqdm import tqdm
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# TODO:
# 1) Check if hard and soft clustering works: CHECK
# 2) Method that performs all steps: CHECK
# 3) Topic reduction: CHECK
# 4) Topic similarity: CHECK
# 5) Word cloud: CHECK
# 6) Sentence embedder for documents: CHECK
# 7) Implement update method: CHECK
# 8) Most similar topics for a word
# 9) Most similar documents for a topic: CHECK
# 10) Topic over time
# 11) Add more tuning parameters to play with: CHECK
# 12) Add other dataset: CHECK
# 13) Implement in streamlit: CHECK
# 14) FIX method create_document_vectors
# 15) Add logging info


def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


class TopicIdentify:

    def __init__(
                self,
                documents,
                embedding_model,
                doc_embedding=None,
                n_neighbors=15,
                n_components=5,
                metric_umap='cosine',
                densmap=False,
                min_cluster_size=15,
                metric_hdbscan='euclidean',
                min_samples=5,
                cluster_selection_method='eom',
                soft_clustering=False,
                cluster_selection_epsilon=0.0,
                save_doc_embed=False,
                path_doc_embed=None,
                lemmatize=False,
                add_stops_words=[],
                min_df=0.005,
                max_df=0.2,
                ngram_range=(1, 3),
                dataset_name="REIT-Industrial",
                random_state=69):

        self.documents = documents
        if doc_embedding is not None:
            self.doc_embedding = self._l2_normalize(doc_embedding)
        else:
            self.doc_embedding = None
        self.embedding_model = embedding_model
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric_umap = metric_umap
        self.densmap = densmap
        self.min_cluster_size = min_cluster_size
        self.metric_hdbscan = metric_hdbscan
        self.cluster_selection_method = cluster_selection_method
        self.soft_clustering = soft_clustering
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.min_samples = min_samples
        self.lemmatize = lemmatize
        self.add_stops_words = add_stops_words
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.save_doc_embed = save_doc_embed
        self.path_doc_embed = path_doc_embed
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.dim_reduction_fit = None
        self.clusterer = None
        self.clusterer_labels = None
        self.clusterer_probs = None
        self.clusterer_cprob = None
        self.topic_sizes = None
        self.topic_sizes_reduced = None
        self.topic_vectors = None
        self.topic_vectors_reduced = None
        self.vocab = None
        self.word_vectors = None
        self.word_indexes = None
        self.topic_words = None
        self.topic_words_reduced = None
        self.topic_word_scores = None
        self.topic_word_scores_reduced = None
        self.topic_hierarchy = None

    def perform_steps(self):
        """
        """
        if self.doc_embedding is None:
            self.create_document_vectors()
        self.dim_reduction()
        self.clustering()
        self.create_topic_vectors()
        self.create_words_vectors()
        self.topic_words, self.topic_word_scores = (
            self.find_topic_words_and_scores(
                self.topic_vectors, self.word_vectors)
        )

    def update(self, step):
        """
        """
        if step == 1:
            if self.doc_embedding is None:
                self.create_document_vectors()
            self.create_words_vectors()
            self.dim_reduction()
            self.clustering()

        elif step == 2:
            self.dim_reduction()
            self.clustering()

        elif step == 3:
            self.clustering()

        self.create_topic_vectors()
        self.create_words_vectors()
        self.topic_words, self.topic_word_scores = (
            self.find_topic_words_and_scores(
                self.topic_vectors, self.word_vectors)
        )

    def dim_reduction(self):
        """
        """

        umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric=self.metric_umap,
            densmap=self.densmap,
            random_state=self.random_state).fit(self.doc_embedding)
        self.dim_reduction_fit = umap_model

    def clustering(self):
        """
        """
        self.random_state
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric_hdbscan,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True).fit(self.dim_reduction_fit.embedding_)
        self.clusterer = clusterer
        self.clusterer_labels = clusterer.labels_
        self.clusterer_cprob = clusterer.probabilities_

        if self.soft_clustering:
            self.clusterer_probs = (
                hdbscan.all_points_membership_vectors(clusterer) /
                (hdbscan.all_points_membership_vectors(clusterer).sum(axis=1).
                    reshape(clusterer.labels_.shape[0], 1))
                )

        if self.soft_clustering:
            self.topic_sizes = (
                    pd.Series(
                        self.clusterer_probs.sum(axis=0)
                    )
            )
        else:
            self.topic_sizes = (
                pd.Series(self.clusterer_labels).value_counts().sort_index()
            )

    def create_topic_vectors(self):
        """
        """
        if self.soft_clustering:
            self.topic_vectors = (
               self._l2_normalize(self.clusterer_probs.T @ self.doc_embedding)
            )
        else:
            unique_labels = set(self.clusterer_labels)
            self.topic_vectors = self._l2_normalize(
                      np.vstack(
                          [self.doc_embedding[np.where(
                              self.clusterer_labels == label)[0]].
                              mean(axis=0) for label in unique_labels]
                      )
                     )

    def create_words_vectors(self):
        """
        """

        all_stopw = text.ENGLISH_STOP_WORDS.union(self.add_stops_words)
        print(all_stopw)
        all_stopw = [i.lower() for i in all_stopw]
        pattern = re.compile(r'\b(' + r'|'.join(all_stopw) + r')\b\s*')
        cleaned_docs = []
        for paragraph in self.documents:
            cleaned_docs.append(pattern.sub('', paragraph.lower()))

        if self.lemmatize:
            vectorizer = CountVectorizer(
                tokenizer=textblob_tokenizer,
                strip_accents='unicode',
                lowercase=True,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range
            )
        else:
            vectorizer = CountVectorizer(
                strip_accents='unicode',
                lowercase=True,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range
            )

        _ = vectorizer.fit_transform(cleaned_docs)
        self.vocab = vectorizer.get_feature_names()

        # embed words
        self.word_indexes = dict(zip(self.vocab, range(len(self.vocab))))
        self.word_vectors = (
            self._l2_normalize(np.array(
                self.embedding_model.encode(self.vocab)))
            )

    def create_document_vectors(self):
        """
        TODO FIX: does not work properly
        """

        batch_size = 500
        document_vectors = []
        current = 0
        batches = int(len(self.documents) / batch_size)
        extra = len(self.documents) % batch_size
        pbar = tqdm(total=batches+1)

        for ind in range(0, batches):
            document_vectors.append(
                self.embedding_model.encode(
                    self.documents[current:current + batch_size])
                )
            pbar.update(1)
            current += batch_size

        if extra > 0:
            document_vectors.append(
                self.embedding_model.encode(
                    self.document[current:current + extra])
            )
        pbar.update(1)
        pbar.close()
        document_vectors = self._l2_normalize(
                np.array(np.vstack(document_vectors))
            )
        if self.save_doc_embed:
            np.save(self.path_doc_embed, document_vectors)

        self.doc_embedding = document_vectors

    @staticmethod
    def _l2_normalize(vectors):
        """
        """
        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]

    def find_topic_words_and_scores(self, topic_vectors, word_vectors):
        """
        """
        topic_words = []
        topic_word_scores = []

        res = np.inner(topic_vectors, word_vectors)
        top_words = np.flip(np.argsort(res, axis=1), axis=1)
        top_scores = np.flip(np.sort(res, axis=1), axis=1)

        for words, scores in zip(top_words, top_scores):
            topic_words.append([self.vocab[i] for i in words[0:50]])
            topic_word_scores.append(scores[0:50])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores

    def topic_similiarity(
            self,
            reduced=False,
            width=750,
            height=750,
            title="Topic similiarity"
            ):

        """
        Given a dataframe containing similarity grid,
        plot an interactive heatmap
        """
        if reduced:
            topic_vectors = self.topic_vectors_reduced
        else:
            topic_vectors = self.topic_vectors
        topic_n_id = {}
        for i1, v1 in enumerate(topic_vectors):
            topic_n_cs = []
            for i2, v2 in enumerate(topic_vectors):
                topic_n_cs.append(
                    cosine_similarity(
                        v1.reshape(1, topic_vectors.shape[1]),
                        v2.reshape(1, topic_vectors.shape[1]))[0][0]
                )
            topic_n_id[i1] = topic_n_cs

        df = pd.DataFrame(topic_n_id).round(2)
        fig = px.imshow(
            df.values,
            labels=dict(x="Topic number", y="Topc number", color="Similarity"),
            x=df.columns,
            y=df.columns,
            color_continuous_scale="Inferno",
            title=title
        )
        fig.update_xaxes(side="bottom")
        fig.update_layout(
            autosize=False,
            width=width,
            height=height
        )
        return fig

    def topic_reduction(self, num_topics):
        """
        Reduce the number of topics discovered by Top2Vec.

        The most representative topics of the corpus will be found, by
        iteratively merging each smallest topic to the most similar topic until
        num_topics is reached.

        Parameters
        ----------
        num_topics: int
            The number of topics to reduce to.

        Returns
        -------
        hierarchy: list of ints
            Each index of hierarchy corresponds to the reduced topics, for each
            reduced topic the indexes of the original topics that were merged
            to create it are listed.

            Example:
            [[3]  <Reduced Topic 0> contains original Topic 3
            [2,4] <Reduced Topic 1> contains original Topics 2 and 4
            [0,1] <Reduced Topic 3> contains original Topics 0 and 1
            ...]
        """

        # skip the noise topic
        top_vecs = self.topic_vectors
        top_sizes = self.topic_sizes.values.tolist()
        num_topics_current = len(top_sizes)
        hierarchy = [[i] for i in range(0, num_topics_current)]
        while num_topics_current > num_topics:

            smallest = np.argmin(top_sizes)
            res = np.inner(top_vecs[smallest], top_vecs)
            sims = np.flip(np.argsort(res))
            most_sim = sims[1]

            # calculate combined topic vector
            top_vec_smallest = top_vecs[smallest]
            smallest_size = top_sizes[smallest]

            top_vec_most_sim = top_vecs[most_sim]
            most_sim_size = top_sizes[most_sim]

            combined_vec = self._l2_normalize(
                ((top_vec_smallest * smallest_size) +
                 ((top_vec_most_sim * most_sim_size)) /
                    (smallest_size + most_sim_size)))

            # update topic vectors
            ix_keep = list(range(len(top_vecs)))
            ix_keep = [i for j, i in enumerate(ix_keep)
                       if j not in [smallest, most_sim]]
            top_vecs = top_vecs[ix_keep]
            top_vecs = np.vstack([top_vecs, combined_vec])
            num_topics_current = top_vecs.shape[0]

            # update sizes
            top_sizes = [i for j, i in enumerate(top_sizes)
                         if j not in [smallest, most_sim]]
            combined_size = smallest_size + most_sim_size
            top_sizes.append(combined_size)

            # update topic hierarchy
            smallest_inds = hierarchy[smallest]
            most_sim_inds = hierarchy[most_sim]
            hierarchy = [i for j, i in enumerate(hierarchy)
                         if j not in [smallest, most_sim]]

            combined_inds = smallest_inds + most_sim_inds
            hierarchy.append(combined_inds)

        self.topic_vectors_reduced = top_vecs
        self.topic_sizes_reduced = top_sizes
        self.topic_hierarchy = [str(i) for i in hierarchy]

        self.topic_words_reduced, self.topic_word_scores_reduced = (
             self.find_topic_words_and_scores(top_vecs, self.word_vectors)
        )

    def search_topic_by_documents(self, topic_nr=1, n=5, reduced=False):
        """
        """
        if reduced:
            topic_vec = self.topic_vectors_reduced
        else:
            topic_vec = self.topic_vectors
        res = np.inner(topic_vec[topic_nr], self.doc_embedding)
        most_similar_idx = np.flip(np.argsort(res))
        return (most_similar_idx[0:n],
                np.take(res, most_similar_idx)[0:n]
                )

    def documents_by_keywords(self, keywords, n=5):
        """
        """
        key_word_embed = self._l2_normalize(
            self.embedding_model.encode(keywords)
        )
        res = np.inner(key_word_embed, self.doc_embedding)
        most_similar_idx = np.flip(np.argsort(res))
        return (most_similar_idx[0:n],
                np.take(res, most_similar_idx)[0:n],
                )

    def generate_topic_wordcloud(
            self,
            topic_num,
            title,
            background_color="white",
            reduced=False):
        """
        Create a word cloud for a topic.

        A word cloud will be generated and displayed. The most semantically
        similar words to the topic will have the largest size, less similar
        words will be smaller. The size is determined using the cosine distance
        of the word vectors from the topic vector.

        Parameters
        ----------
        topic_num: int
            The topic number to search.

        background_color : str (Optional, default='white')
            Background color for the word cloud image. Suggested options are:
                * white
                * black

        reduced: bool (Optional, default False)
            Original topics are used by default. If True the
            reduced topics will be used.

        Returns
        -------
        A matplotlib plot of the word cloud with the topic number will be
        displayed.

        """

        if reduced:
            word_score_dict = dict(
                zip(self.topic_words_reduced[topic_num],
                    self.topic_word_scores_reduced[topic_num]))
        else:
            word_score_dict = dict(
                zip(self.topic_words[topic_num],
                    self.topic_word_scores[topic_num]))

        plt.figure(figsize=(10, 8), dpi=200)
        plt.axis("off")
        plt.imshow(
            WordCloud(
                width=600,
                height=400,
                background_color=background_color,
                random_state=69).generate_from_frequencies(word_score_dict))
        plt.title("Topic " + str(title),
                  loc='left', fontsize=20, pad=20)
