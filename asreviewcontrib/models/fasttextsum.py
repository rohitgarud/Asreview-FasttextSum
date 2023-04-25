from math import log2

import numpy as np
import pandas as pd

try:
    from gensim.models.fasttext import FastText as GenSimFastText
    from gensim.utils import simple_preprocess

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

from asreview.models.feature_extraction.base import BaseFeatureExtraction
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _check_gensim():
    if not GENSIM_AVAILABLE:
        raise ImportError("Install gensim package to use" " FastText.")


class FasttextSum(BaseFeatureExtraction):
    name = "fasttext_sum"

    def __init__(
        self,
        *args,
        vector_size=40,
        epochs=33,
        min_count=1,
        window=7,
        n_jobs=1,
        pooling="mean",
        ngram_max=1,
        stop_words=None,  # "english"
        sif_a=1e-3,  # Used if pooling is SIF
        **kwargs
    ):
        super(FasttextSum, self).__init__(*args, **kwargs)
        self.vector_size = int(vector_size)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.window = int(window)
        self.n_jobs = int(n_jobs)
        self.pooling = pooling
        self._model = None
        # Used if pooling is tfidf
        self.ngram_max = ngram_max
        self.stop_words = stop_words
        self.sif_a = sif_a  # Used if pooling is SIF
        if stop_words is None or stop_words.lower() == "none":
            sklearn_stop_words = None
        else:
            sklearn_stop_words = self.stop_words

        if self.pooling in ["tf", "log_tf", "sif"]:
            self.countvec = CountVectorizer(stop_words=sklearn_stop_words)
        elif self.pooling in ["mean", "tfidf"]:
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, ngram_max), stop_words=sklearn_stop_words
            )

    def fit(self, texts):

        # check if gensim is available
        _check_gensim()

        model_param = {
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "window": self.window,
            "workers": self.n_jobs,
        }

        corpus = [simple_preprocess(text) for text in texts]

        self._model = GenSimFastText(corpus, **model_param)

        if self.pooling in ["tf", "log_tf", "sif"]:
            self.countvec.fit(texts)
        elif self.pooling in ["mean", "tfidf"]:
            self.tfidf.fit(texts)

    def transform(self, texts):

        # check if gensim is available
        _check_gensim()

        corpus = [simple_preprocess(text) for text in texts]

        X = np.zeros((len(texts), self.vector_size))
        if self.pooling in ["tf", "log_tf", "sif"]:
            self.count_features = self.countvec.transform(texts).toarray()
            self.word_count = np.sum(self.count_features, axis=0)
            self.total_words = sum(self.word_count)
            self.word_count_map = {
                word: self.word_count[idx]
                for word, idx in self.countvec.vocabulary_.items()
            }

        elif self.pooling in ["mean", "tfidf"]:
            self.tfidf_features = self.tfidf.transform(texts).toarray()

        for doc_id, doc_text in enumerate(corpus):
            doc_vec = np.zeros(self.vector_size).reshape(1, -1)
            if len(doc_text) > 0:
                num_vecs = 0
                for word in doc_text:
                    weight = self._get_weight(word, doc_id)
                    num_vecs += 1 if weight > 0 else 0
                    doc_vec += weight * self._model.wv[word].reshape(1, -1)
                if self.pooling in ["mean", "sif"] and num_vecs > 0:
                    doc_vec /= num_vecs
            X[doc_id] = doc_vec

        if self.pooling == "sif":
            svd = TruncatedSVD(
                n_components=1, n_iter=7, random_state=0
            )  # Add global random state
            svd.fit(X)
            pc = svd.components_  # Principal components
            X = X - X.dot(pc.T) * pc
            return X
        else:
            return X

    def _get_weight(self, word, doc_id):
        if self.pooling in ["tf", "log_tf", "sif"]:
            idx = self.countvec.vocabulary_.get(word)
        elif self.pooling in ["mean", "tfidf"]:
            idx = self.tfidf.vocabulary_.get(word)

        if self.pooling == "mean":
            weight = 1 if idx else 0
        elif self.pooling == "tfidf":
            weight = self.tfidf_features[doc_id][idx] if idx else 0
        elif self.pooling == "tf":
            weight = self.count_features[doc_id][idx] if idx else 0
        elif self.pooling == "log_tf":
            weight = log2(self.count_features[doc_id][idx] + 1) if idx else 0
        elif self.pooling == "sif":
            wc = self.word_count_map.get(word)
            if wc:
                p_w = wc / self.total_words  # Unigram probability
                weight = self.sif_a / (self.sif_a + p_w)
            else:
                weight = 0

        return weight
