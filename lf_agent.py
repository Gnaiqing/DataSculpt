import sklearn
import numpy as np
import os
from numpy.random import default_rng
from lf_family import KeywordLF
from data_utils import TextDataset, TextPairDataset


def get_lf_agent(train_dataset, valid_dataset, agent_type, **kwargs):
    if agent_type == "simulated":
        return SimLFAgent(train_dataset, valid_dataset, **kwargs)
    else:
        raise ValueError("LF agent not supported.")


class SentimentLexicon:
    def __init__(self, data_root):
        pos_words_file = os.path.join(data_root, 'opinion-lexicon-English/positive-words.txt')
        neg_words_file = os.path.join(data_root, 'opinion-lexicon-English/negative-words.txt')

        pos_tokens = list()
        neg_tokens = list()
        with open(pos_words_file, encoding='ISO-8859-1') as f:
            for i, line in enumerate(f):
                if i >= 30:
                    token = line.rstrip()
                    pos_tokens.append(token)
        with open(neg_words_file, encoding='ISO-8859-1') as f:
            for i, line in enumerate(f):
                if i >= 31:
                    token = line.rstrip()
                    neg_tokens.append(token)

        token_sentiment = {token: 1 for token in pos_tokens}
        token_sentiment.update({token: -1 for token in neg_tokens})

        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
        self.token_sentiment = token_sentiment

    def tokens_to_sentiments(self, tokens):
        """Return sentiments of tokens in a sentence
        """
        sentiments = np.array([self.token_sentiment.get(token, 0) for token in tokens])

        return sentiments

    def tokens_with_sentiment(self, tokens, sentiment):
        """Return tokens with specified sentiment
        """
        sentiments = self.tokens_to_sentiments(tokens)
        tokens = np.array(tokens)[sentiments == sentiment]

        return tokens


class SimLFAgent:
    def __init__(self, train_dataset, valid_dataset, lf_type="keyword", filter_methods=("acc","unique", "consist"),
                 acc_threshold=0.6, data_root="./data", seed=0, **kwargs):
        """
        Simulated LF Agent that return a LF with accuracy above threshold
        :param train_dataset: unlabeled training set
        :param valid_dataset: validation set
        :param lf_type: type of LF that the agent returns
        :param filter_methods:
                "acc": LF accuracy is above threshold
                "sentiment" : the keyword has corresponding sentiment
                "unique" : the LF was not returned in previous iterations
                "consist" : the LF is accurate on corresponding development instance
        :param acc_threshold: threshold for LF accuracy
        :param data_root:
        :param repeat: whether the agent may return the same LF multiple times
        :param seed: random seed
        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lf_type = lf_type
        self.filter_methods = filter_methods
        if "sentiment" in self.filter_methods:
            self.sentiment_lexicon = SentimentLexicon(data_root)

        self.lfs = list() # history LFs
        self.acc_threshold = acc_threshold
        self.rng = default_rng(seed)

    def create_lf(self, query_idx):
        item = self.train_dataset[query_idx]
        candidate_lfs = []
        if self.lf_type == "keyword":
            for label in self.train_dataset.classes:
                for token in item["token"]:
                    lf = KeywordLF(keyword=token, label=label)
                    candidate_lfs.append(lf)
        else:
            raise ValueError ("LF Type not supported.")

        mask = np.repeat(True, len(candidate_lfs))
        if "consist" in self.filter_methods:
            for j, lf in enumerate(candidate_lfs):
                if lf.label != item["label"]:
                    mask[j] = False

        if "acc" in self.filter_methods:
            for j, lf in enumerate(candidate_lfs):
                if not mask[j]:
                    continue
                cov, acc = lf.get_cov_acc(self.train_dataset)
                if acc < self.acc_threshold:
                    mask[j] = False

        if "sentiment" in self.filter_methods:
            for j, lf in enumerate(candidate_lfs):
                if not mask[j]:
                    continue
                if lf.label == 0:
                    tokens = self.sentiment_lexicon.neg_tokens
                else:
                    tokens = self.sentiment_lexicon.pos_tokens
                if lf.keyword not in tokens:
                    mask[j] = False

        if "unique" in self.filter_methods:
            for j, lf in enumerate(candidate_lfs):
                if not mask[j]:
                    continue
                for prev_lf in self.lfs:
                    if lf == prev_lf:
                        mask[j] = False
                        break

        filtered_lfs = np.array(candidate_lfs)[mask]
        if len(filtered_lfs) > 0:
            lf = self.rng.choice(filtered_lfs)
            self.lfs.append(lf)
        else:
            lf = None

        return lf





