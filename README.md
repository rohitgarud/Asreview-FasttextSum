# ASReview FasttextSum Extension
This extension adds a feature extraction technique that creates features by a weighted sum of the Fasttext word embeddings to generate document embeddings. Different weighting schemes are implemented.

## Getting started
To install this extension, clone the repository to your system and then run the following command from inside the repository.

```bash
pip install .
```

or we can directly install it from GitHub using

```bash
pip install git+https://github.com/rohitgarud/Asreview-FasttextSum.git
```

## Usage
This extension adds a feature extraction method called `fasttext_sum`. Four different weighting schemes (pooling methods) are currently available: `mean`, `tfidf`(Term Frequency - Inverse Document Frequency Weighed), `tf`(Term Frequency Weighted), `log_tf`(log2 scaled Term Frequency Weighted). Simulations can be performed using the simulation mode from ASReview CLI using:

```bash
asreview simulate benchmark:van_de_Schoot_2017 -m logistic -e fasttext_sum
```
The default is the `mean` pooling with a 40-dimensional feature vector. To use other settings, we have to use the Python API. As the features can contain negative values, the Naive Bayes classifier cannot be used with features generated using FasttextSum. 

## License

Apache 2.0 license
