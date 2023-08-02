import os
import sys
import json
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import argparse
import pdb

hub_data_paths = ["glue"]

feature_name_dict = {
    "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2")
}

def create_bert_vector(raw_texts, save_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(raw_texts)
    np.save(save_path, embeddings)

    return embeddings


def tr_val_te_split(xs, ys, test_ratio, valid_ratio, rand_state):
    xs = np.array(xs)
    ys = np.array(ys)
    assert len(xs) == len(ys)
    N = len(xs)

    permuted_idxs = rand_state.permutation(N)
    num_test = int(N * test_ratio)
    num_valid = int(N * valid_ratio)

    train_idxs, test_idxs = permuted_idxs[:-num_test], permuted_idxs[-num_test:]
    # num_valid = int(len(train_idxs) * valid_ratio)
    train_idxs, valid_idxs = train_idxs[:-num_valid], train_idxs[-num_valid:]

    return (xs[train_idxs], ys[train_idxs], xs[valid_idxs], ys[valid_idxs],
            xs[test_idxs], ys[test_idxs], train_idxs, valid_idxs, test_idxs)


def load_local_data(data_root, dataset_name, feature, test_ratio=0.1, valid_ratio=0.1, warmup_ratio=0.0,
                    rand_state=np.random.RandomState(0)):
    # load raw sentences and labels
    if dataset_name == 'AmazonReview':
        dataset = AmazonReviewDataset(data_root=os.path.join(data_root, 'AmazonReview'))
    elif dataset_name == 'IMDB':
        dataset = IMDBDataset(data_root=os.path.join(data_root, 'aclImdb'))
    elif dataset_name == 'SST':
        dataset = SSTDataset(data_root=os.path.join(data_root, 'SST-2'))
    elif dataset_name == 'Yelp':
        dataset = YelpDataset(data_root=os.path.join(data_root, 'yelp_review_polarity_csv'))
    elif dataset_name == 'sms':
        dataset = SMSDataset(data_root=os.path.join(data_root, 'sms'))
    elif dataset_name == 'bios':
        dataset = BiosDataset(data_root=os.path.join(data_root, 'bios'))
    elif dataset_name == 'agnews':
        dataset = AGNewsDataset(data_root=os.path.join(data_root, 'agnews'), rand_state=rand_state)
    elif dataset_name == 'yahoo':
        dataset = YahooDataset(data_root=os.path.join(data_root, 'yahoo'), rand_state=rand_state)
    elif dataset_name == 'youtube':
        dataset = YoutubeDataset(data_root=os.path.join(data_root, 'spam/data'))
    else:
        raise ValueError('Dataset not supported.')

    raw_texts = dataset.raw_texts
    labels = dataset.labels

    (xs_text_tr, ys_tr, xs_text_val, ys_val,
     xs_text_te, ys_te, train_idxs, valid_idxs, test_idxs) = tr_val_te_split(raw_texts, labels, test_ratio, valid_ratio,
                                                                             rand_state)

    # create tokenized texts for LF labeling use
    count_vectorizer = CountVectorizer(strip_accents='ascii')
    count_vectorizer.fit(xs_text_tr)
    vocab = count_vectorizer.vocabulary_
    analyzer = count_vectorizer.build_analyzer()
    xs_token_tr = np.array([analyzer(text) for text in xs_text_tr], dtype='object')
    xs_token_val = np.array([analyzer(text) for text in xs_text_val], dtype='object')
    xs_token_te = np.array([analyzer(text) for text in xs_text_te], dtype='object')

    # create features (independent of the above tokenization process)
    if feature == 'tfidf':
        tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_df=0.9, max_features=1000)
        xs_feature_tr = tfidf_vectorizer.fit_transform(xs_text_tr).toarray()
        xs_feature_val = tfidf_vectorizer.transform(xs_text_val).toarray()
        xs_feature_te = tfidf_vectorizer.transform(xs_text_te).toarray()

        scaler = StandardScaler()
        xs_feature_tr = scaler.fit_transform(xs_feature_tr)
        xs_feature_val = scaler.transform(xs_feature_val)
        xs_feature_te = scaler.transform(xs_feature_te)

    elif feature == 'embedding':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), analyzer='word')
        xs_feature_tr = vectorizer.fit_transform(xs_text_tr)
        xs_feature_val = vectorizer.transform(xs_text_val)
        xs_feature_te = vectorizer.transform(xs_text_te)

        n, m = xs_feature_tr.shape

        svd = TruncatedSVD(n_components=500, n_iter=5, random_state=0)
        xs_feature_tr = svd.fit_transform(xs_feature_tr).astype(float)
        xs_feature_val = svd.transform(xs_feature_val).astype(float)
        xs_feature_te = svd.transform(xs_feature_te).astype(float)

        # scaler = StandardScaler()
        # xs_feature_tr = scaler.fit_transform(xs_feature_tr)
        # xs_feature_val = scaler.transform(xs_feature_val)
        # xs_feature_te = scaler.transform(xs_feature_te)

    elif feature == 'bert':
        saved_file = os.path.join(data_root, 'embeddings', dataset_name, 'bert.npy')
        if os.path.exists(saved_file):
            embeddings = np.load(saved_file)
        else:
            save_path = os.path.join(data_root, 'embeddings', dataset_name, 'bert.npy')
            embeddings = create_bert_vector(raw_texts, save_path)

        xs_feature_tr = embeddings[train_idxs]
        xs_feature_val = embeddings[valid_idxs]
        xs_feature_te = embeddings[test_idxs]

    else:
        raise ValueError('Feature representation not supported.')

    num_train = len(ys_tr)
    if warmup_ratio > 1:
        num_warmup = int(warmup_ratio)
    else:
        num_warmup = int(num_train * warmup_ratio)

    permuted_idxs = rand_state.permutation(num_train)
    warmup_idxs, train_idxs = permuted_idxs[:num_warmup], permuted_idxs[num_warmup:]

    xs_text_wu, xs_token_wu, xs_feature_wu, ys_wu = xs_text_tr[warmup_idxs], xs_token_tr[warmup_idxs], xs_feature_tr[
        warmup_idxs], ys_tr[warmup_idxs]
    xs_text_tr, xs_token_tr, xs_feature_tr, ys_tr = xs_text_tr[train_idxs], xs_token_tr[train_idxs], xs_feature_tr[
        train_idxs], ys_tr[train_idxs]

    train_dataset = TextDataset(xs_text_tr, xs_token_tr, xs_feature_tr, ys_tr, vocab)
    valid_dataset = TextDataset(xs_text_val, xs_token_val, xs_feature_val, ys_val, vocab, classes=train_dataset.classes)
    test_dataset = TextDataset(xs_text_te, xs_token_te, xs_feature_te, ys_te, vocab, classes=train_dataset.classes)
    warmup_dataset = TextDataset(xs_text_wu, xs_token_wu, xs_feature_wu, ys_wu, vocab, classes=train_dataset.classes)
    train_dataset.convert_neg_zero_label()
    valid_dataset.convert_neg_zero_label()
    test_dataset.convert_neg_zero_label()
    warmup_dataset.convert_neg_zero_label()

    return train_dataset, valid_dataset, test_dataset, warmup_dataset


def extract_text_labels(raw_dataset, feature_name=("sentence",), label_name="label"):
    """
    Extract texts and labels from raw dataset
    :param raw_dataset: raw dataset downloaded from hub
    :param feature_name: tuple of feature names
    :param label_name: column for label
    :return: raw_text_1, raw_text_2 (for sentence pairs), labels
    """
    for feature in feature_name:
        assert feature in raw_dataset.features

    assert label_name in raw_dataset.features

    raw_text_1 = np.array(raw_dataset[feature_name[0]])
    if len(feature_name) > 1:
        raw_text_2 = np.array(raw_dataset[feature_name[1]])
    else:
        raw_text_2 = None

    labels = np.array(raw_dataset[label_name])
    return raw_text_1, raw_text_2, labels



def load_hub_data(dataset_path, dataset_name, feature, test_ratio=0.1, valid_ratio=0.1, warmup_ratio=0.0,
                  rand_state=np.random.RandomState(0), resplit=False, data_root="./data"):
    raw_datasets = load_dataset(dataset_path, dataset_name)
    feature_name = feature_name_dict[dataset_name]
    if len(feature_name) > 1:
        pair_cls = True
    else:
        pair_cls = False

    raw_text_1_tr, raw_text_2_tr, labels_tr = extract_text_labels(raw_datasets["train"], feature_name=feature_name)
    raw_text_1_val, raw_text_2_val, labels_val = extract_text_labels(raw_datasets["validation"], feature_name=feature_name)
    raw_text_1_te, raw_text_2_te, labels_te = extract_text_labels(raw_datasets["test"], feature_name=feature_name)
    if resplit:
        raw_text_1 = np.concatenate((raw_text_1_tr,raw_text_1_val,raw_text_1_te))
        labels = np.concatenate((labels_tr, labels_val, labels_te))
        (xs_text_tr, ys_tr, xs_text_val, ys_val,
         xs_text_te, ys_te, train_idxs, valid_idxs, test_idxs) = tr_val_te_split(raw_text_1, labels, test_ratio,
                                                                                 valid_ratio,
                                                                                 rand_state)

        if pair_cls:
            raw_text_2 = np.concatenate((raw_text_2_tr, raw_text_2_val, raw_text_2_te))
            xs_text_2_tr = raw_text_2[train_idxs]
            xs_text_2_val = raw_text_2[valid_idxs]
            xs_text_2_te = raw_text_2[test_idxs]
    else:
        raw_text_1 = np.concatenate((raw_text_1_tr,raw_text_1_val,raw_text_1_te))
        train_idxs = np.arange(len(raw_text_1_tr))
        valid_idxs = np.arange(len(raw_text_1_val)) + len(raw_text_1_tr)
        test_idxs = np.arange(len(raw_text_1_te)) + len(raw_text_1_tr) + len(raw_text_1_val)
        xs_text_tr, xs_text_val, xs_text_te = raw_text_1_tr, raw_text_1_val, raw_text_1_te
        ys_tr, ys_val, ys_te = labels_tr, labels_val, labels_te
        if pair_cls:
            raw_text_2 = np.concatenate((raw_text_2_tr, raw_text_2_val, raw_text_2_te))
            xs_text_2_tr, xs_text_2_val, xs_text_2_te = raw_text_2_tr, raw_text_2_val, raw_text_2_te


    # create tokenized texts for LF labeling
    count_vectorizer = CountVectorizer(strip_accents='ascii')
    if pair_cls:
        count_vectorizer.fit(np.concatenate((xs_text_tr, xs_text_2_tr)))
    else:
        count_vectorizer.fit(xs_text_tr)

    vocab = count_vectorizer.vocabulary_
    analyzer = count_vectorizer.build_analyzer()
    xs_token_tr = np.array([analyzer(text) for text in xs_text_tr], dtype='object')
    xs_token_val = np.array([analyzer(text) for text in xs_text_val], dtype='object')
    xs_token_te = np.array([analyzer(text) for text in xs_text_te], dtype='object')

    if pair_cls:
        xs_token_2_tr = np.array([analyzer(text) for text in xs_text_2_tr], dtype='object')
        xs_token_2_val = np.array([analyzer(text) for text in xs_text_2_val], dtype='object')
        xs_token_2_te = np.array([analyzer(text) for text in xs_text_2_te], dtype='object')

    # create features (independent of the above tokenization process)
    if feature == 'tfidf':
        tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_df=0.9,
                                           max_features=1000)
        if pair_cls:
            tfidf_vectorizer.fit(np.concatenate((xs_text_tr, xs_text_2_tr)))
        else:
            tfidf_vectorizer.fit(xs_text_tr)

        xs_feature_tr = tfidf_vectorizer.transform(xs_text_tr).toarray()
        xs_feature_val = tfidf_vectorizer.transform(xs_text_val).toarray()
        xs_feature_te = tfidf_vectorizer.transform(xs_text_te).toarray()

        if pair_cls:
            xs_feature_2_tr = tfidf_vectorizer.transform(xs_text_2_tr).toarray()
            xs_feature_2_val = tfidf_vectorizer.transform(xs_text_2_val).toarray()
            xs_feature_2_te = tfidf_vectorizer.transform(xs_text_2_te).toarray()

        scaler = StandardScaler()
        xs_feature_tr = scaler.fit_transform(xs_feature_tr)
        xs_feature_val = scaler.transform(xs_feature_val)
        xs_feature_te = scaler.transform(xs_feature_te)

        if pair_cls:
            scaler = StandardScaler()
            xs_feature_2_tr = scaler.fit_transform(xs_feature_2_tr)
            xs_feature_2_val = scaler.transform(xs_feature_2_val)
            xs_feature_2_te = scaler.transform(xs_feature_2_te)

    elif feature == 'embedding':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1),
                                     analyzer='word')
        if pair_cls:
            vectorizer.fit(np.concatenate((xs_text_tr, xs_text_2_tr)))
        else:
            vectorizer.fit(xs_text_tr)

        xs_feature_tr = vectorizer.transform(xs_text_tr)
        xs_feature_val = vectorizer.transform(xs_text_val)
        xs_feature_te = vectorizer.transform(xs_text_te)
        if pair_cls:
            xs_feature_2_tr = vectorizer.transform(xs_text_2_tr).toarray()
            xs_feature_2_val = vectorizer.transform(xs_text_2_val).toarray()
            xs_feature_2_te = vectorizer.transform(xs_text_2_te).toarray()

        n, m = xs_feature_tr.shape

        svd = TruncatedSVD(n_components=500, n_iter=5, random_state=0)
        xs_feature_tr = svd.fit_transform(xs_feature_tr).astype(float)
        xs_feature_val = svd.transform(xs_feature_val).astype(float)
        xs_feature_te = svd.transform(xs_feature_te).astype(float)
        if pair_cls:
            svd = TruncatedSVD(n_components=500, n_iter=5, random_state=0)
            xs_feature_2_tr = svd.fit_transform(xs_feature_2_tr).astype(float)
            xs_feature_2_val = svd.transform(xs_feature_2_val).astype(float)
            xs_feature_2_te = svd.transform(xs_feature_2_te).astype(float)

        # scaler = StandardScaler()
        # xs_feature_tr = scaler.fit_transform(xs_feature_tr)
        # xs_feature_val = scaler.transform(xs_feature_val)
        # xs_feature_te = scaler.transform(xs_feature_te)

    elif feature == 'bert':
        saved_file = os.path.join(data_root, 'embeddings', dataset_name, 'bert.npy')
        if os.path.exists(saved_file):
            embeddings = np.load(saved_file)
        else:
            save_path = os.path.join(data_root, 'embeddings', dataset_name, 'bert.npy')
            embeddings = create_bert_vector(raw_text_1, save_path)

        xs_feature_tr = embeddings[train_idxs]
        xs_feature_val = embeddings[valid_idxs]
        xs_feature_te = embeddings[test_idxs]
        if pair_cls:
            saved_file = os.path.join(data_root, 'embeddings', dataset_name, '_2_bert.npy')
            if os.path.exists(saved_file):
                embeddings = np.load(saved_file)
            else:
                save_path = os.path.join(data_root, 'embeddings', dataset_name, '_2_bert.npy')
                embeddings = create_bert_vector(raw_text_2, save_path)

            xs_feature_2_tr = embeddings[train_idxs]
            xs_feature_2_val = embeddings[valid_idxs]
            xs_feature_2_te = embeddings[test_idxs]

    else:
        raise ValueError('Feature representation not supported.')

    num_train = len(ys_tr)
    if warmup_ratio > 1:
        num_warmup = int(warmup_ratio)
    else:
        num_warmup = int(num_train * warmup_ratio)

    permuted_idxs = rand_state.permutation(num_train)
    warmup_idxs, train_idxs = permuted_idxs[:num_warmup], permuted_idxs[num_warmup:]

    if pair_cls:
        xs_text_wu, xs_token_wu, xs_feature_wu, ys_wu, xs_text_2_wu, xs_token_2_wu, xs_feature_2_wu = \
            (xs_text_tr[warmup_idxs], xs_token_tr[warmup_idxs],xs_feature_tr[warmup_idxs], ys_tr[warmup_idxs],
             xs_text_2_tr[warmup_idxs], xs_token_2_tr[warmup_idxs], xs_feature_2_tr[warmup_idxs])
        xs_text_tr, xs_token_tr, xs_feature_tr, ys_tr, xs_text_2_tr, xs_token_2_tr, xs_feature_2_tr = \
            (xs_text_tr[train_idxs], xs_token_tr[train_idxs],xs_feature_tr[train_idxs], ys_tr[train_idxs],
             xs_text_2_tr[train_idxs], xs_token_2_tr[train_idxs], xs_feature_2_tr[train_idxs])

        train_dataset = TextPairDataset(xs_text_tr, xs_token_tr, xs_feature_tr,xs_text_2_tr, xs_token_2_tr,
                                        xs_feature_2_tr, ys_tr, vocab)
        valid_dataset = TextPairDataset(xs_text_val, xs_token_val, xs_feature_val, xs_text_2_val, xs_token_2_val,
                                        xs_feature_2_val, ys_val, vocab, classes=train_dataset.classes)
        test_dataset = TextPairDataset(xs_text_te, xs_token_te, xs_feature_te, xs_text_2_te, xs_token_2_te,
                                       xs_feature_2_te, ys_te, vocab, classes=train_dataset.classes)
        warmup_dataset = TextPairDataset(xs_text_wu, xs_token_wu, xs_feature_wu, xs_text_2_wu, xs_token_2_wu,
                                         xs_feature_2_wu, ys_wu, vocab, classes=train_dataset.classes)


    else:
        xs_text_wu, xs_token_wu, xs_feature_wu, ys_wu = xs_text_tr[warmup_idxs], xs_token_tr[warmup_idxs], xs_feature_tr[
            warmup_idxs], ys_tr[warmup_idxs]
        xs_text_tr, xs_token_tr, xs_feature_tr, ys_tr = xs_text_tr[train_idxs], xs_token_tr[train_idxs], xs_feature_tr[
            train_idxs], ys_tr[train_idxs]

        train_dataset = TextDataset(xs_text_tr, xs_token_tr, xs_feature_tr, ys_tr, vocab)
        valid_dataset = TextDataset(xs_text_val, xs_token_val, xs_feature_val, ys_val, vocab, classes=train_dataset.classes)
        test_dataset = TextDataset(xs_text_te, xs_token_te, xs_feature_te, ys_te, vocab, classes=train_dataset.classes)
        warmup_dataset = TextDataset(xs_text_wu, xs_token_wu, xs_feature_wu, ys_wu, vocab, classes=train_dataset.classes)

    return train_dataset, valid_dataset, test_dataset, warmup_dataset


class TextDataset:
    def __init__(self, xs_text, xs_token, xs_feature, ys, vocab, classes=None):
        assert np.all(np.array([len(xs_text), len(xs_token), len(xs_feature)]) == len(ys))
        self.xs_text = xs_text
        self.xs_token = xs_token
        self.xs_feature = xs_feature
        self.ys = ys
        if classes is None:
            self.classes = np.unique(self.ys)
            self.n_class = len(self.classes)
        else:
            self.classes = classes
            self.n_class = len(self.classes)

        self.vocab = vocab  # vocabulary that map terms to feature indices
        self.revert_index = {}
        for idx in range(len(self.xs_token)):
            for token in self.xs_token[idx]:
                if token in self.revert_index:
                    self.revert_index[token].append(idx)
                else:
                    self.revert_index[token] = [idx]

        for token in self.revert_index:
            self.revert_index[token] = np.array(self.revert_index[token])



    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        item = {
            "sentence": self.xs_text[idx],
            "token": self.xs_token[idx],
            "feature": self.xs_feature[idx],
            "idx": idx,
            "label": self.ys[idx]
        }
        return item

    def convert_neg_zero_label(self):
        zero_pos = self.ys == 0
        neg_pos = self.ys == -1
        self.ys[zero_pos] = -1
        self.ys[neg_pos] = 0
        if -1 in self.classes:
            pos = np.where(self.classes == -1)
            self.classes[pos] = 0

    def apply_lfs(self, lfs):
        weak_labels = []
        for lf in lfs:
            wl = lf.apply_to_dataset(self)
            weak_labels.append(wl.reshape(-1, 1))

        weak_labels = np.hstack(weak_labels)
        return weak_labels


class TextPairDataset(TextDataset):
    def __init__(self, xs_text_1, xs_token_1, xs_feature_1, xs_text_2, xs_token_2, xs_feature_2, ys, vocab, classes=None):
        super(TextPairDataset, self).__init__(xs_text_1, xs_token_1, xs_feature_1, ys, vocab, classes=classes)
        assert np.all(np.array([len(xs_text_2), len(xs_token_2), len(xs_feature_2)]) == len(ys))
        self.xs_text_2 = xs_text_2
        self.xs_token_2 = xs_token_2
        self.xs_feature_2 = xs_feature_2
        self.revert_index_2 = {}
        for idx in range(len(self.xs_token_2)):
            for token in self.xs_token_2[idx]:
                if token in self.revert_index_2:
                    self.revert_index_2[token].append(idx)
                else:
                    self.revert_index_2[token] = [idx]

        for token in self.revert_index_2:
            self.revert_index_2[token] = np.array(self.revert_index_2[token])

    def __getitem__(self, idx):
        item = {
            "sentence": self.xs_text[idx] + " [SEP] "+ self.xs_text_2[idx],   # for ease of printing
            "sentence1": self.xs_text[idx],
            "token1": self.xs_token[idx],
            "feature1": self.xs_feature[idx],
            "sentence2": self.xs_text_2[idx],
            "token2": self.xs_token_2[idx],
            "feature2": self.xs_feature_2[idx],
            "idx": idx,
            "label": self.ys[idx]
        }
        return item


class SSTDataset:
    def __init__(self, data_root, subsample_size=20000):
        self.data_root = data_root
        self.subsample_size = subsample_size
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.tsv')
        with open(data_file) as f:
            lines = f.readlines()
        lines = lines[1:]  # drop the headers
        lines = [line.rstrip().split('\t') for line in lines]
        lines = np.array(lines, 'object')
        # subsample
        lines = lines[np.random.permutation(len(lines))[:self.subsample_size]]
        raw_texts = lines[:, 0]
        labels = lines[:, 1].astype(int)
        labels = labels * 2 - 1

        return raw_texts, labels


class SMSDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'spam.csv')
        df = pd.read_csv(data_file, sep=',', header=0, encoding='latin-1').to_numpy()[:, :2]

        raw_texts = df[:, 1]
        labels = df[:, 0]
        labels[labels == 'ham'] = -1
        labels[labels == 'spam'] = 1
        labels = labels.astype(int)

        return raw_texts, labels


class BiosDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'professor_teacher.csv')
        # data_file = os.path.join(self.data_root, 'painter_architect.csv')
        # data_file = os.path.join(self.data_root, 'journalist_photographer.csv')
        df = pd.read_csv(data_file, sep=',', header=0).to_numpy()[:, :2]

        raw_texts = df[:, 0]
        labels = df[:, 1]
        labels = labels.astype(int)

        return raw_texts, labels


class YoutubeDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        files = ['Youtube01-Psy.csv', 'Youtube02-KatyPerry.csv', 'Youtube03-LMFAO.csv',
                 'Youtube04-Eminem.csv', 'Youtube05-Shakira.csv']
        df_all = None
        for f in files:
            data_file = os.path.join(self.data_root, f)
            df = pd.read_csv(data_file, sep=',', header=0)
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])

        df = df_all.to_numpy()
        raw_texts = df[:, 3]
        labels = df[:, 4]
        labels = labels.astype(int)
        labels[labels == 0] = -1

        return raw_texts, labels


class AGNewsDataset:
    def __init__(self, data_root, rand_state):
        self.data_root = data_root
        self.rand_state = rand_state
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.csv')
        df = pd.read_csv(data_file, sep=',', header=0).to_numpy()

        raw_texts = df[:, 2]
        labels = df[:, 0].astype(int)

        class_0 = 3
        class_1 = 4

        sport_idx = labels == class_0
        sci_idx = labels == class_1

        raw_texts_sport = raw_texts[sport_idx]
        labels_sport = labels[sport_idx]

        raw_texts_sci = raw_texts[sci_idx]
        labels_sci = labels[sci_idx]

        raw_texts_all = np.hstack([raw_texts_sport, raw_texts_sci])
        labels_all = np.hstack([labels_sport, labels_sci])

        idx_permutation = self.rand_state.permutation(len(labels_all))
        raw_texts = raw_texts_all[idx_permutation]
        labels = labels_all[idx_permutation]
        labels[labels == class_0] = -1
        labels[labels == class_1] = 1

        return raw_texts, labels


class YahooDataset:
    def __init__(self, data_root, rand_state):
        self.data_root = data_root
        self.rand_state = rand_state
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.csv')
        df = pd.read_csv(data_file, sep=',', header=None)

        df[4] = df[1].astype(str) + ' ' + df[2].astype(str) + ' ' + df[3].astype(str)

        df = df.to_numpy()

        raw_texts = df[:, 4]
        labels = df[:, 0].astype(int)

        class_0 = 2
        class_1 = 6

        class_0_idx = labels == class_0
        class_1_idx = labels == class_1

        raw_texts_0 = raw_texts[class_0_idx]
        labels_0 = labels[class_0_idx]

        raw_texts_1 = raw_texts[class_1_idx]
        labels_1 = labels[class_1_idx]

        raw_texts_all = np.hstack([raw_texts_0, raw_texts_1])
        labels_all = np.hstack([labels_0, labels_1])

        idx_permutation = self.rand_state.permutation(len(labels_all))[:30000]
        raw_texts = raw_texts_all[idx_permutation]
        labels = labels_all[idx_permutation]
        labels[labels == class_0] = -1
        labels[labels == class_1] = 1

        return raw_texts, labels


class YelpDataset:
    def __init__(self, data_root, subsample_size=25000):
        self.data_root = data_root
        self.subsample_size = subsample_size
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.csv')
        df = pd.read_csv(data_file, sep=',', header=None).to_numpy()
        # subsample
        df_subsampled = df[np.random.permutation(len(df))[:self.subsample_size]]
        raw_texts = df_subsampled[:, 1]
        labels = df_subsampled[:, 0].astype(int)
        labels = (labels - 1) * 2 - 1

        return raw_texts, labels


class AmazonReviewDataset:
    """ AmazonReview Dataset for Sentiment Analysis (Polarity)
    """

    def __init__(self, data_root, subsample_size=500, pos_rating_threshold=5, neg_rating_threshold=1,
                 saved_path=None, random_state=None):
        self.data_root = data_root
        self.subsample_size = subsample_size
        self.pos_rating_threshold = pos_rating_threshold
        self.neg_rating_threshold = neg_rating_threshold
        self.saved_path = saved_path
        self.random_state = random_state

        # create dataset with raw text
        df = None
        if saved_path is None:
            # print('looking for "amazon_review.pkl"...')
            if os.path.exists(f'{data_root}/amazon_review.pkl'):
                # print('loading raw text dataset from {}'.format('"./amazon_review.pkl"'))
                df = pd.read_pickle(f'{data_root}/amazon_review.pkl')
            else:
                print('building raw text dataset...')
                df = self.build_dataset()
                df.to_pickle('./amazon_review.pkl')
        else:
            print('loading raw text dataset...')
            df = pd.read_pickle(saved_path)

        # extract raw setences and labels
        self.df = df
        self.raw_texts = df['reviewText'].to_numpy()
        self.labels = df['sentiment'].to_numpy()

    def parse_file(self, path, category):
        print('Processing {}...'.format(category))
        data = list()
        with gzip.open(path) as f:
            for line in f:
                record = json.loads(line)
                # only keep records with rating above/below specified thresholds
                if (record['overall'] >= self.pos_rating_threshold or
                        record['overall'] <= self.neg_rating_threshold):
                    record['category'] = category
                    data.append(record)
        df = pd.DataFrame.from_records(data)

        # keep only relevant features
        df = df[['reviewText', 'category', 'overall']]

        # partition reviews into positive/negative sentiments
        df_pos = df[df['overall'] >= self.pos_rating_threshold]
        df_neg = df[df['overall'] <= self.neg_rating_threshold]

        # subsample data from positive/negative subsets
        df_pos = df_pos.sample(n=self.subsample_size, replace=False, random_state=self.random_state)
        df_pos['sentiment'] = 1
        df_neg = df_neg.sample(n=self.subsample_size, replace=False, random_state=self.random_state)
        df_neg['sentiment'] = -1
        df = pd.concat([df_pos, df_neg], axis=0)

        return df

    def build_dataset(self):
        files = os.listdir(self.data_root)
        paths = [os.path.join(self.data_root, file) for file in files]
        categories = [file.split('.')[0].lower()[8:-2] for file in files]
        print('Categories: {}'.format(categories))

        dfs = list()
        for path, category in tqdm(zip(paths, categories), total=len(paths)):
            df = self.parse_file(path, category)
            if df is not None:
                dfs.append(df)
        df = pd.concat(dfs, axis=0)

        return df


class IMDBDataset:
    def __init__(self, data_root):
        self.data_root = os.path.join(data_root, 'train')
        self.pos_dir = os.path.join(self.data_root, 'pos')
        self.neg_dir = os.path.join(self.data_root, 'neg')

        raw_texts, labels = self.build_dataset()

        self.raw_texts = np.array(raw_texts, dtype='object')
        self.labels = np.array(labels)

    def build_dataset(self):
        pos_files = os.listdir(self.pos_dir)
        pos_paths = [os.path.join(self.pos_dir, file) for file in pos_files]
        pos_texts = list()
        for path in pos_paths:
            with open(path) as f:
                line = f.readline().rstrip()
                pos_texts.append(line)

        neg_files = os.listdir(self.neg_dir)
        neg_paths = [os.path.join(self.neg_dir, file) for file in neg_files]
        neg_texts = list()
        for path in neg_paths:
            with open(path) as f:
                line = f.readline().rstrip()
                neg_texts.append(line)

        raw_texts = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [-1] * len(neg_texts)

        return raw_texts, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="glue")
    parser.add_argument("--dataset-name", type=str, default="sst2")
    parser.add_argument("--feature-extractor",type=str, default="tfidf")
    args = parser.parse_args()
    if args.dataset_path != "glue":
        train_dataset, valid_dataset, test_dataset, warmup_dataset = load_local_data(args.dataset_path,
                                                                                     args.dataset_name, args.feature_extractor, 0.1, 0.1, 0., np.random.RandomState(0))
    else:
        train_dataset, valid_dataset, test_dataset, warmup_dataset = load_hub_data(args.dataset_path,
                args.dataset_name, args.feature_extractor, rand_state=np.random.RandomState(0))


    print("Train size:", len(train_dataset))
    print("Valid size:", len(valid_dataset))
    print("Test size:", len(test_dataset))