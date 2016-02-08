import numpy as np
from itertools import tee

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from gensim.models import Word2Vec

from urlembed.util.seqmanager import tokenize_and_stem

class Url2Vec:

    # Constructor
    def __init__ (self, codeurl_map):
        # assert (type(codeurl_map) is dict), "url2vec needs a map that associates a code(e.g. a number) for each URL"
        self.labels_ = None
        # assume is list-like or fail gracefully
        self.codeurl_map = codeurl_map if type(codeurl_map) is dict else { str(x): codeurl_map[x] for x in range(len(codeurl_map)) }


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
    def __word_embedding(self, sequences_list, vecs_length=48):
        assert hasattr(sequences_list, '__iter__'), "Bad sequences argument"

        w2v_model = Word2Vec(min_count=1, negative=5, size=vecs_length)
        build_seq, train_seq = tee(sequences_list)
        w2v_model.build_vocab(build_seq)
        w2v_model.train(train_seq)
        return np.array([w2v_model[code] for code in self.codeurl_map])


    # returns tfidf vector for each page
    def __tfidf(self, codecontent_map, vecs_length=50, tfidf=True):
        assert set(codecontent_map.keys()) == set(self.codeurl_map.keys()), "NEIN"

        self.codecontent_map = codecontent_map
        self.pages_content = [self.codecontent_map[code] for code in self.codeurl_map]
        self.codes = [code for code in self.codeurl_map]
        self.longurls = [self.codeurl_map[code] for code in self.codeurl_map]

        tfidf_vectorizer = TfidfVectorizer(
            max_df = 0.9,
            max_features = 200000,
            min_df = 0.05,
            stop_words = 'english',
            use_idf = tfidf,
            tokenizer = tokenize_and_stem, # to fix
            ngram_range = (1, 3)
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.pages_content)
        svd = TruncatedSVD(n_components=vecs_length, algorithm="arpack", random_state=1)
        return svd.fit_transform(tfidf_matrix)


    # calls the chosen algorithm with the data builded from the input arguments
    def train(self, algorithm=HDBSCAN(min_cluster_size=7), use_w2v=True, use_tfidf=True,
              w2v_size=48, tfidf_size=50, sequences_list=None, codecontent_map=None):

        assert (use_w2v and sequences_list is not None or use_tfidf and codecontent_map is not None), "Bad arguments!"

        empty_array = np.array([ [] for i in range(len(self.codeurl_map)) ])
        w2v_vecs    = self.__word_embedding(sequences_list, vecs_length=w2v_size) if use_w2v else empty_array
        tfidf_vecs  = self.__tfidf(codecontent_map, vecs_length=tfidf_size) if use_tfidf else empty_array

        data = [ np.append(w2v_vecs[i], tfidf_vecs[i]) for i in range(len(self.codeurl_map)) ]
        self.labels_ = algorithm.fit_predict(data)
        self.labels_ = [int(x) for x in self.labels_] # map(int, self.labels_)
        return self.labels_


    # needs the real membership (ground truth) and the membership returned by the algorithm (pred_membership)
    # ...(already given if train was successful)
    # returns the confusion matrix
    def test(self, ground_truth, pred_membership=None):
        assert (pred_membership is not None or self.labels_ is not None), "No train, No test !"
        pred_membership = self.labels_ if pred_membership is None else pred_membership
        self.ground_truth = ground_truth

        return self.__get_confusion_table(ground_truth, pred_membership)
