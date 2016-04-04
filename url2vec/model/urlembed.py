__author__ = 'chris'

import numpy as np
from itertools import tee

from sklearn import metrics
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

from url2vec.util.plotter import scatter_plot
from sklearn.decomposition import TruncatedSVD
from url2vec.util.seqmanager import tokenize_and_stem, get_color
from sklearn.feature_extraction.text import TfidfVectorizer

class Url2Vec:

    # Constructor
    def __init__ (self, codeurl_map):
        # assert (type(codeurl_map) is dict), "url2vec needs a map that associates a code(e.g. a number) for each URL"
        self.labels_  = None
        self.training = None

        if type(codeurl_map) is dict:
            self.codeurl_map = codeurl_map
            self.urls  = [self.codeurl_map[code] for code in self.codeurl_map]
            self.codes = [code for code in self.codeurl_map]
        else:
            self.urls = codeurl_map
            self.codeurl_map = { str(x): codeurl_map[x] for x in range(len(codeurl_map)) }
            self.codes = [ str(x) for x in range(len(codeurl_map)) ]


    # matching matrix
    def __get_confusion_table(self, ground_truth, predicted_labels):
        assert len(ground_truth) == len(predicted_labels), "Invalid input arguments"
        assert len(ground_truth) > 0, "Invalid input arguments"
        assert isinstance(ground_truth[0], int), "Type is not int"
        assert isinstance(predicted_labels[0], int), "Type is not int"

        # matrix(ground_truth x predicted_labels)
        conf_table = np.zeros((len(set(ground_truth)), len(set(predicted_labels))))
        real_clust = list(set(ground_truth))
        # it's needed because ground truth can have discontinuous cluster set
        clust_to_index = { real_clust[i]: i for i in range(len(real_clust)) }

        for real_clust in clust_to_index.values():
            for i in range(len(predicted_labels)):
                if clust_to_index[ground_truth[i]] == real_clust:
                    cluster_found = predicted_labels[i]
                    conf_table[real_clust, cluster_found] = conf_table[real_clust, cluster_found] + 1
        return conf_table


    # trains word2vec with the given parameters and returns vectors for each page
    def __word_embedding(self, sequences, sg=0, min_count=1, window=10, negative=5, size=48):
        assert (hasattr(sequences, '__iter__')), "Bad sequences argument"
        build_seq, train_seq, assert_seq = tee(sequences, 3)
        first = next( (word for sequence in assert_seq for word in sequence if word is not None), None)
        assert (first is not None), "Empty sequences"

        w2v_model = Word2Vec(sg=sg, min_count=min_count, window=window, negative=negative, size=size)
        build_seq, train_seq = tee(sequences)
        w2v_model.build_vocab(build_seq)
        w2v_model.train(train_seq)
        urlvecs = []
        if first in set(self.codes):
            urlvecs = np.array([w2v_model[code] for code in self.codes])
        elif first in set(self.urls):
            urlvecs = np.array([w2v_model[self.codeurl_map[code]] for code in self.codeurl_map])
        else:
            raise AttributeError("sequences don't match with URLs")
        return urlvecs


    # returns tfidf vector for each page
    def __tfidf(self, contents, max_df=0.9, max_features=200000, min_df=0.05, dim_red=50, tfidf=True):

        if type(contents) is dict:
            assert ( set(contents.keys()) == set(self.codeurl_map.keys()) ), "NEIN"
            self.contents = contents
            self.pages_content = [self.contents[code] for code in self.codes]
        else:
            self.contents = {str(x): contents[x] for x in range(len(contents))}
            self.pages_content = [self.contents[code] for code in self.codes]

        tfidf_vectorizer = TfidfVectorizer(
            max_df = max_df,
            max_features = max_features,
            min_df = min_df,
            stop_words = 'english',
            use_idf = tfidf,
            tokenizer = tokenize_and_stem, # to fix
            ngram_range = (1, 3)
        )

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.pages_content)
        svd = TruncatedSVD(n_components= dim_red, algorithm="arpack", random_state=1)
        return svd.fit_transform(tfidf_matrix)


    # calls the chosen algorithm with the data builded from the input arguments
    def train(self, algorithm=HDBSCAN(min_cluster_size=7), use_w2v=True, use_tfidf=True,
              e_sg=0, e_min_count=1, e_window=10, e_negative=5, e_size=48, sequences=None,
              d_max_df=0.9, d_max_features=200000, d_min_df=0.05, d_dim_red=50, d_tfidf=True, contents=None):

        assert (use_w2v and sequences is not None or use_tfidf and contents is not None), "Bad arguments!"

        empty_array = np.array([ [] for i in range(len(self.codeurl_map)) ])
        w2v_vecs    = self.__word_embedding(
            sequences,
            sg=e_sg,
            min_count=e_min_count,
            window=e_window,
            negative=e_negative,
            size=e_size
        ) if use_w2v else empty_array

        tfidf_vecs  = self.__tfidf(
            contents,
            max_df=d_max_df,
            max_features=d_max_features,
            min_df=d_min_df,
            dim_red=d_dim_red,
            tfidf=d_tfidf
        ) if use_tfidf else empty_array

        self.training = [ np.append(w2v_vecs[i], tfidf_vecs[i]) for i in range(len(self.codeurl_map)) ]
        self.labels_ = algorithm.fit_predict(self.training)
        self.labels_ = [int(x) for x in self.labels_]
        return self.labels_


    # needs the real membership (ground truth) and the membership returned by the algorithm (pred_membership)
    # ...(already given if train was successful)
    # returns the confusion matrix
    def test(self, ground_truth, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No train, No test !"
        pred_membership = self.labels_ if pred_membership is None else pred_membership
        self.ground_truth = ground_truth

        return self.__get_confusion_table(ground_truth, pred_membership)

    # homogeneity score
    def homogeneity_score(self, ground_truth=None, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No prediction yet !"
        assert (ground_truth is not None or self.ground_truth is not None), "Ground Truth not given"

        pred_membership = self.labels_ if pred_membership is None else pred_membership
        ground_truth = self.ground_truth if ground_truth is None else ground_truth

        return metrics.homogeneity_score(ground_truth, pred_membership)

    # completeness score
    def completeness_score(self, ground_truth=None, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No prediction yet !"
        assert (ground_truth is not None or self.ground_truth is not None), "Ground Truth not given"

        pred_membership = self.labels_ if pred_membership is None else pred_membership
        ground_truth = self.ground_truth if ground_truth is None else ground_truth

        return metrics.completeness_score(ground_truth, pred_membership)

    # v-measure score
    def v_measure_score(self, ground_truth=None, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No prediction yet !"
        assert (ground_truth is not None or self.ground_truth is not None), "Ground Truth not given"

        pred_membership = self.labels_ if pred_membership is None else pred_membership
        ground_truth = self.ground_truth if ground_truth is None else ground_truth

        return metrics.v_measure_score(ground_truth, pred_membership)

    # adjusted rand score
    def adjusted_rand_score(self, ground_truth=None, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No prediction yet !"
        assert (ground_truth is not None or self.ground_truth is not None), "Ground Truth not given"

        pred_membership = self.labels_ if pred_membership is None else pred_membership
        ground_truth = self.ground_truth if ground_truth is None else ground_truth

        return metrics.adjusted_rand_score(ground_truth, pred_membership)

    # adjusted mutual information
    def adjusted_mutual_info_score(self, ground_truth=None, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No prediction yet !"
        assert (ground_truth is not None or self.ground_truth is not None), "Ground Truth not given"

        pred_membership = self.labels_ if pred_membership is None else pred_membership
        ground_truth = self.ground_truth if ground_truth is None else ground_truth

        return metrics.adjusted_mutual_info_score(ground_truth, pred_membership)

    # silhouette score
    def silhouette_score(self, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No prediction yet !"
        assert (self.training is not None), "No training yet !"

        pred_membership = self.labels_ if pred_membership is None else pred_membership

        return metrics.silhouette_score(np.array(self.training), np.array(pred_membership), metric='euclidean')

    # uses t-SNE for dimensionality redustion
    def two_dim(self, high_dim_vecs=None):
        assert (high_dim_vecs is not None or self.training is not None), "No prediction yet !"
        high_dim_vecs = self.training if high_dim_vecs is None else high_dim_vecs
        tsne = TSNE(n_components=2)

        self.twodim = tsne.fit_transform(high_dim_vecs)

        return self.twodim

    # plots data
    def plot_trace(self, twodim=None, urls=None, pred_membership=None, user="chrispolo", api_key="89nned6csl"):
        assert (twodim is not None or self.twodim is not None), "No twodim vectors !"
        assert (urls is not None or self.urls is not None), "No urls !"
        assert (pred_membership is not None or self.labels_ is not None), "No prediction yet !"

        twodim = self.twodim if twodim is None else twodim
        urls = self.urls if urls is None else urls
        pred_membership = self.labels_ if pred_membership is None else pred_membership

        return scatter_plot(twodim, urls, [get_color(clust) for clust in pred_membership], user, api_key)
