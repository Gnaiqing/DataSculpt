import sklearn
import numpy as np
import os
from numpy.random import default_rng
from lf_family import KeywordLF
import openai
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


def extract_label_keywords(content, cardinality=2):
    """
    Extract label and keywords from contents returned by LLMs
    :param content: returned contents
    :return: label, keyword_list
    """
    tokens = nltk.word_tokenize(content)
    classes = range(cardinality)
    if "LABEL" not in tokens or "KEYWORDS" not in tokens:
        return None, None

    label_idx = tokens.index("LABEL") + 2
    if label_idx < len(tokens) and tokens[label_idx].isdigit() and int(tokens[label_idx]) in classes:
        label = int(tokens[label_idx])
    else:
        return None, None

    keyword_idx = tokens.index("KEYWORDS") + 2
    if "EXPLANATION" in tokens:
        last_keyword_pos = tokens.index("EXPLANATION")
    else:
        last_keyword_pos = len(tokens)
    keyword_list = []
    for idx in np.arange(last_keyword_pos - keyword_idx) + keyword_idx:
        keyword = tokens[idx]
        if keyword != "NA":
            keyword_list.append(keyword.lower())

    return label, keyword_list



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
                 acc_threshold=0.6, data_root="./data", seed=0, model="gpt-3.5-turbo", api_key_path="openai-api.key",
                 display=True, repeats=1, prompt_version="v1", **kwargs):
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
        self.display = display
        self.repeats = repeats
        self.prompt_version = prompt_version
        self.kwargs = kwargs
        openai.api_key_path = api_key_path

    def create_lf(self, query_idx):
        item = self.train_dataset.examples[query_idx]["text"]
        candidate_lfs = []
        if self.lf_type == "keyword":
            system_prompt = get_system_prompt(self.kwargs["dataset_name"], prompt_version=self.prompt_version)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item}
            ]
            for i in range(self.repeats): # try multiple times if first attempt fails
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages = messages
                    )
                    response_content = response['choices'][0]["message"]["content"]
                except openai.error.OpenAIError:
                    response_content = ""

                if self.display:
                    print("Response: ", response_content)

                label, keyword_list = extract_label_keywords(response_content, self.train_dataset.n_class)
                if label is not None:
                    break

            if label is not None:
                for keyword in keyword_list:
                    if keyword in item:
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
            lf = KeywordLF(keyword="NA", label=label)

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
        item = self.train_dataset.examples[query_idx]["text"]
        label = self.train_dataset.labels[query_idx]
        tokens = nltk.word_tokenize(item.lower())
        candidate_lfs = []
        if self.lf_type == "keyword":
            for token in tokens:
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
            lf = KeywordLF(keyword="NA", label=label)

        return lf





