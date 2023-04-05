from math import log2

import numpy as np

try:
    from gensim.models.fasttext import FastText as GenSimFastText
    from gensim.utils import simple_preprocess

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

from asreview.models.feature_extraction.base import BaseFeatureExtraction
from sklearn.feature_extraction.text import TfidfVectorizer


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
        **kwargs
    ):
        super(FastText, self).__init__(*args, **kwargs)
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
        if stop_words is None or stop_words.lower() == "none":
            sklearn_stop_words = None
        else:
            sklearn_stop_words = self.stop_words

        if self.pooling in ["tf", "log_tf"]:
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, ngram_max),
                stop_words=sklearn_stop_words,
                sublinear_tf=False,
                binary=False,
                use_idf=False,
                smooth_idf=False,
            )
        else:
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, ngram_max), stop_words=sklearn_stop_words
            )

    def fit(self, texts):

        # check is gensim is available
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
        self.tfidf.fit(texts)

    def transform(self, texts):

        # check is gensim is available
        _check_gensim()

        corpus = [simple_preprocess(text) for text in texts]

        X = []
        if self.pooling in ["tfidf", "tf", "log_tf"]:
            tfidf_features = self.tfidf.transform(texts).toarray()

        for i, doc_text in enumerate(corpus):
            doc_vec = np.zeros(self.vector_size).reshape(1, -1)
            if len(doc_text) > 0:
                total_weight = 0
                for word in doc_text:
                    idx = self.tfidf.vocabulary_.get(word)

                    if self.pooling == "mean":
                        weight = 1 if idx else 0
                    elif self.pooling in ["tfidf", "tf"]:
                        weight = tfidf_features[i][idx] if idx else 0
                    elif self.pooling in "log_tf":
                        weight = log2(tfidf_features[i][idx] + 1) if idx else 0

                    total_weight += weight
                    doc_vec += weight * self._model.wv[word].reshape(1, -1)
                doc_vec /= total_weight
            X.append(doc_vec)
        return np.array(X).reshape(len(texts), self.vector_size)
