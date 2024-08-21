import re
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from wrench.dataset import load_dataset, BaseDataset
from pathlib import Path
import os
import json


def preprocess_text(text, stop_words=None, stemming="porter"):
    if stop_words is not None:
        stop_words = set(stopwords.words(stop_words))
    else:
        stop_words = set()

    if stemming == "porter":
        stemmer = PorterStemmer()
    elif stemming == "snowball":
        stemmer = SnowballStemmer(language="english")
    else:
        stemmer = None

    processed_tokens = []
    tokens = nltk.word_tokenize(text.lower())
    for token in tokens:
        # filter out stopwords and non-words
        if token in stop_words or (re.search("^\w+$", token) is None):
            continue
        if stemmer is not None:
            token = stemmer.stem(token)

        processed_tokens.append(token)

    processed_text = " ".join(processed_tokens)
    return processed_text


def build_revert_index(dataset, stop_words=None, stemming="porter", max_ngram=1, cache_path=None):
    """
    Build reverted index for the given text dataset
    """
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path) as infile:
                reverted_index = json.load(infile)
                for phrase in reverted_index:
                    reverted_index[phrase] = np.array(reverted_index[phrase])

                return reverted_index

    # preprocess data
    corpus = [dataset.examples[idx]["text"] for idx in range(len(dataset))]
    if stop_words is not None:
        stop_words = set(stopwords.words(stop_words))
    else:
        stop_words = set()

    if stemming == "porter":
        stemmer = PorterStemmer()
    elif stemming == "snowball":
        stemmer = SnowballStemmer(language="english")
    else:
        stemmer = None

    reverted_index = {}

    for idx, text in enumerate(corpus):
        processed_tokens = []
        text = re.sub("\ufeff", "", text)
        tokens = nltk.word_tokenize(text.lower())
        for token in tokens:
            # filter out stopwords and non-words
            if token in stop_words or (re.search("^\w+$", token) is None):
                continue
            if stemmer is not None:
                token = stemmer.stem(token)

            processed_tokens.append(token)

        for n in range(max_ngram):
            phrases = ngrams(processed_tokens, n+1)
            for t in phrases:
                phrase = " ".join(t)
                if phrase in reverted_index:
                    if reverted_index[phrase][-1] != idx:
                        reverted_index[phrase].append(idx)
                else:
                    reverted_index[phrase] = [idx]

    if cache_path is not None:
        with open(cache_path, "w") as outfile:
            json.dump(reverted_index, outfile)

    for phrase in reverted_index:
        reverted_index[phrase] = np.array(reverted_index[phrase])

    return reverted_index


def load_wrench_data(data_root, dataset_name, feature, stopwords=None, stemming="porter", max_ngram=1, revert_index=True):
    if feature in ["tfidf", "bow"]:
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn=feature)
    else:
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn=feature, cache_name=feature)


    if revert_index:
        train_cache_path = Path(data_root) / dataset_name / "train_index.json"
        valid_cache_path = Path(data_root) / dataset_name / "valid_index.json"
        test_cache_path = Path(data_root) / dataset_name / "test_index.json"
        train_dataset.revert_index = build_revert_index(train_dataset, stop_words=stopwords, stemming=stemming,
                                                        max_ngram=max_ngram, cache_path=train_cache_path)
        valid_dataset.revert_index = build_revert_index(valid_dataset, stop_words=stopwords, stemming=stemming,
                                                        max_ngram=max_ngram, cache_path=valid_cache_path)
        test_dataset.revert_index = build_revert_index(test_dataset, stop_words=stopwords, stemming=stemming,
                                                       max_ngram=max_ngram, cache_path=test_cache_path)

    return train_dataset, valid_dataset, test_dataset