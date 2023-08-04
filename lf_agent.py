import sklearn
import numpy as np
import os
from numpy.random import default_rng
from lf_family import KeywordLF
import openai
from data_utils import TextDataset, TextPairDataset
from gpt_utils import get_system_prompt
import nltk

def get_lf_agent(train_dataset, valid_dataset, agent_type, **kwargs):
    if agent_type == "simulated":
        return SimLFAgent(train_dataset, valid_dataset, **kwargs)
    elif agent_type == "chatgpt":
        return ChatGPTLFAgent(train_dataset, valid_dataset, **kwargs)
    else:
        raise ValueError("LF agent not supported.")



def filter_candidate_lfs(candidate_lfs, filter_methods, valid_dataset,
                         acc_threshold, sentiment_lexicon, prev_lfs):
    """
    Filter a subset of candidate lfs that can be added given previous LFs

    :param candidate_lfs: candidate LFs for the current iteration
    :param filter_methods: methods for filtering
    :param valid_dataset: validation dataset for estimating LF accuracy
    :param acc_threshold: accuracy threshold for accuracy filter
    :param sentiment_lexicon: sentiment lexicons for sentiment filter
    :param prev_lfs: LFs designed in previous iterations for redundance filter
    :return: filtered_lfs: a subset of candidate LFs
    """
    mask = np.repeat(True, len(candidate_lfs))

    if "acc" in filter_methods:
        for j, lf in enumerate(candidate_lfs):
            if not mask[j]:
                continue
            cov, acc = lf.get_cov_acc(valid_dataset)
            if np.isnan(acc) or acc < acc_threshold:
                mask[j] = False

    if "sentiment" in filter_methods:
        for j, lf in enumerate(candidate_lfs):
            if not mask[j]:
                continue
            if lf.label == 0:
                tokens = sentiment_lexicon.neg_tokens
            else:
                tokens = sentiment_lexicon.pos_tokens
            if lf.keyword not in tokens:
                mask[j] = False

    if "unique" in filter_methods:
        for j, lf in enumerate(candidate_lfs):
            if not mask[j]:
                continue
            for prev_lf in prev_lfs:
                if lf == prev_lf:
                    mask[j] = False
                    break

    filtered_lfs = np.array(candidate_lfs)[mask]
    return filtered_lfs


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


class ChatGPTLFAgent:
    def __init__(self, train_dataset, valid_dataset, lf_type="keyword", filter_methods=("acc","unique"),
                 acc_threshold=0.6, data_root="./data", seed=0, model="gpt-3.5-turbo", api_key_path="openai-api.key",**kwargs):
        """
        LF Agent using ChatGPT
        :param train_dataset:
        :param valid_dataset:
        :param lf_type:
        :param filter_methods:
        :param acc_threshold:
        :param data_root:
        :param seed:
        :param kwargs:
        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lf_type = lf_type
        self.filter_methods = filter_methods
        self.sentiment_lexicon = SentimentLexicon(data_root)
        self.lfs = list()  # history LFs
        self.acc_threshold = acc_threshold
        self.rng = default_rng(seed)
        self.model = model
        self.kwargs = kwargs
        openai.api_key_path = api_key_path

    def create_lf(self, query_idx):
        item = self.train_dataset[query_idx]
        candidate_lfs = []
        if self.lf_type == "keyword":
            system_prompt = get_system_prompt(self.train_dataset.dataset_name)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["sentence"]}
            ]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages = messages
            )
            response_content = response['choices'][0]["message"]["content"]
            response_tokens = nltk.word_tokenize(response_content)
            if "LABEL" not in response_tokens or "KEYWORDS" not in response_tokens:
                return None

            label_idx = response_tokens.index("LABEL") + 2
            label = int(response_tokens[label_idx])
            keyword_idx = response_tokens.index("KEYWORDS") + 2
            for idx in np.arange(len(response_tokens)-keyword_idx) + keyword_idx:
                keyword = response_tokens[idx]
                if keyword in item["token"]:
                    lf = KeywordLF(keyword=keyword, label=label)
                    candidate_lfs.append(lf)

        else:
            raise ValueError ("LF Type not supported.")

        filtered_lfs = filter_candidate_lfs(candidate_lfs, self.filter_methods, self.valid_dataset,
                                            self.acc_threshold, self.sentiment_lexicon, self.lfs)

        if len(filtered_lfs) > 0:
            lf = self.rng.choice(filtered_lfs)
            self.lfs.append(lf)
        else:
            lf = None

        return lf





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
        self.sentiment_lexicon = SentimentLexicon(data_root)
        self.lfs = list() # history LFs
        self.acc_threshold = acc_threshold
        self.rng = default_rng(seed)

    def create_lf(self, query_idx):
        item = self.train_dataset[query_idx]
        label = item["label"]
        candidate_lfs = []
        if self.lf_type == "keyword":
            for token in item["token"]:
                lf = KeywordLF(keyword=token, label=label)
                candidate_lfs.append(lf)
        else:
            raise ValueError ("LF Type not supported.")

        filtered_lfs = filter_candidate_lfs(candidate_lfs, self.filter_methods, self.valid_dataset,self.acc_threshold,
                                            self.sentiment_lexicon, self.lfs)

        if len(filtered_lfs) > 0:
            lf = self.rng.choice(filtered_lfs)
            self.lfs.append(lf)
        else:
            lf = None

        return lf





