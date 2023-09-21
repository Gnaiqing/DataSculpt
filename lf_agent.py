import sklearn
import numpy as np
import os
from numpy.random import default_rng
from lf_family import KeywordLF
import openai
from gpt_utils import create_prompt
from data_utils import relation_extraction_datasets, preprocess_text
import nltk
import re


def get_lf_agent(train_dataset, valid_dataset, agent_type, **kwargs):
    if agent_type == "simulated":
        return SimLFAgent(train_dataset, valid_dataset, **kwargs)
    elif agent_type == "chatgpt":
        return ChatGPTLFAgent(train_dataset, valid_dataset, **kwargs)
    else:
        raise ValueError("LF agent not supported.")


def filter_candidate_lfs(candidate_lfs, filter_methods, train_dataset,
                         valid_dataset, acc_threshold, prev_lfs):
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

    for j, lf in enumerate(candidate_lfs):
        train_cov, _ = lf.get_cov_acc(train_dataset)
        if train_cov == 0:
            mask[j] = False

    if "acc" in filter_methods:
        for j, lf in enumerate(candidate_lfs):
            if not mask[j]:
                continue
            cov, acc = lf.get_cov_acc(valid_dataset)
            if cov > 0 and acc < acc_threshold:
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
    label_match = re.search("LABEL:\s*\d+", content)
    if label_match:
        st, ed = label_match.span()
        label = int(label_match.string[st+6:ed])
        if label not in range(cardinality):
            label = None
    else:
        label = None

    keyword_match = re.search("KEYWORDS:.*\n", content)
    if keyword_match:
        st, ed = keyword_match.span()
        keyword_list = [x.strip() for x in keyword_match.string[st+9:ed].split(',')]

    else:
        keyword_list = []

    return label, keyword_list


class ChatGPTLFAgent:
    def __init__(self, train_dataset, valid_dataset, **kwargs):
        """
        LF Agent using ChatGPT
        :param train_dataset: training dataset to label
        :param valid_dataset: validation dataset
        :param kwargs: arguments
        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.kwargs = kwargs
        # API related arguments
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        openai.api_key_path = kwargs.get("api_key_path", "openai-api.key")
        self.repeats = kwargs.get("repeats", 1)
        self.example_per_class = kwargs.get("example_per_class", 1)
        self.example_selection = kwargs.get("example_selection", "random")
        self.return_explanation = kwargs.get("return_explanation", False)
        self.play_expert_role = kwargs.get("play_expert_role", False)
        self.dp_aware = kwargs.get("dp_aware", False)
        self.n_completion = kwargs.get("n_completion")
        self.temperature = kwargs.get("temperature")
        self.top_p = kwargs.get("top_p")
        # Label function related arguments
        self.lf_type = kwargs.get("lf_type", "keyword")
        self.filter_methods = kwargs.get("filter_methods", ("acc", "unique"))
        self.lfs = list()  # history LFs
        self.acc_threshold = kwargs.get("acc_threshold", 0.6)
        self.stop_words = kwargs.get("stop_words")
        self.stemming = kwargs.get("stemming")
        self.max_ngram = kwargs.get("max_ngram", 1)
        self.max_lf_per_iter = kwargs.get("max_lf_per_iter", 1)
        # other arguments
        self.display = kwargs.get("display", True)
        self.seed = kwargs.get("seed", 0)
        self.rng = default_rng(self.seed)
        self.system_prompt, self.example_prompt = create_prompt(self.kwargs["dataset_name"], self.valid_dataset,
                                                                example_per_class=self.example_per_class,
                                                                example_selection=self.example_selection,
                                                                explanation=self.return_explanation,
                                                                expert_role=self.play_expert_role,
                                                                dp_aware=self.dp_aware)
        if self.display:
            print("ChatGPT system prompt:")
            print(self.system_prompt)
            print("Example prompt:")
            print(self.example_prompt)

    def create_lf(self, query_idx):
        if self.kwargs["dataset_name"] == "cdr":
            text = self.train_dataset.examples[query_idx]["text"]
            entity1 = self.train_dataset.examples[query_idx]["entity1"]
            entity2 = self.train_dataset.examples[query_idx]["entity2"]
            user_prompt = "{} User: {}. Does {} cause{}?\n Response: ".format(self.example_prompt, text, entity1, entity2)
        elif self.kwargs["dataset_name"] == "chemprot":
            text = self.train_dataset.examples[query_idx]["text"]
            entity1 = self.train_dataset.examples[query_idx]["entity1"]
            entity2 = self.train_dataset.examples[query_idx]["entity2"]
            user_prompt = "{} User: {}. What is the relationship between {} and {}?\n Response: ".format(self.example_prompt,
                                                                                                        text, entity1, entity2)
        else:
            text = self.train_dataset.examples[query_idx]["text"]
            user_prompt = "{} User: {}\n Response: ".format(self.example_prompt, text)

        candidate_lfs = []
        if self.lf_type == "keyword":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            label, keyword_list = None, []
            for i in range(self.repeats):  # try multiple times if first attempt fails
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        n=self.n_completion
                    )
                except openai.error.OpenAIError:
                    continue

                output_labels = []
                for j in range(self.n_completion):
                    response_content = response['choices'][j]["message"]["content"]
                    if self.display:
                        print("Response {}: {}\n".format(j, response_content))
                    label, keywords = extract_label_keywords(response_content, self.train_dataset.n_class)
                    if label in range(self.train_dataset.n_class):
                        output_labels.append(label)
                    keyword_list += keywords

                if len(output_labels) > 0:
                    label = np.bincount(output_labels).argmax()
                    break

            keyword_list = np.unique(keyword_list)
            if label is not None:
                for keyword in keyword_list:
                    processed_keyword = preprocess_text(keyword, self.stop_words, self.stemming)
                    n_gram = len(processed_keyword.split(" "))
                    if n_gram <= self.max_ngram:
                        lf = KeywordLF(keyword=processed_keyword, label=label)
                        candidate_lfs.append(lf)

        else:
            raise ValueError("LF Type not supported.")

        filtered_lfs = filter_candidate_lfs(candidate_lfs, self.filter_methods, self.train_dataset,
                                            self.valid_dataset, self.acc_threshold, self.lfs)

        if self.max_lf_per_iter is not None and self.max_lf_per_iter < len(filtered_lfs):
            filtered_lfs = self.rng.choice(filtered_lfs, size=self.max_lf_per_iter, replace=False)

        if len(filtered_lfs) == 0:
            filtered_lfs = [KeywordLF("NA", label)]
        else:
            for lf in filtered_lfs:
                self.lfs.append(lf)

        return filtered_lfs


class SimLFAgent:
    def __init__(self, train_dataset, valid_dataset, lf_type="keyword", filter_methods=("acc","unique"),
                 acc_threshold=0.6, seed=0, **kwargs):
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
        self.lfs = list() # history LFs
        self.acc_threshold = acc_threshold
        self.rng = default_rng(seed)
        self.max_lf_per_iter = kwargs.get("max_lf_per_iter", 1)

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

        filtered_lfs = filter_candidate_lfs(candidate_lfs, self.filter_methods, self.train_dataset, self.valid_dataset,
                                            self.acc_threshold, self.lfs)

        if self.max_lf_per_iter is not None and self.max_lf_per_iter < len(filtered_lfs):
            filtered_lfs = self.rng.choice(filtered_lfs, size=self.max_lf_per_iter, replace=False)

        if len(filtered_lfs) == 0:
            filtered_lfs = [KeywordLF("NA", label)]
        else:
            for lf in filtered_lfs:
                self.lfs.append(lf)

        return filtered_lfs


# class SentimentLexicon:
#     def __init__(self, data_root):
#         pos_words_file = os.path.join(data_root, 'opinion-lexicon-English/positive-words.txt')
#         neg_words_file = os.path.join(data_root, 'opinion-lexicon-English/negative-words.txt')
#
#         pos_tokens = list()
#         neg_tokens = list()
#         with open(pos_words_file, encoding='ISO-8859-1') as f:
#             for i, line in enumerate(f):
#                 if i >= 30:
#                     token = line.rstrip()
#                     pos_tokens.append(token)
#         with open(neg_words_file, encoding='ISO-8859-1') as f:
#             for i, line in enumerate(f):
#                 if i >= 31:
#                     token = line.rstrip()
#                     neg_tokens.append(token)
#
#         token_sentiment = {token: 1 for token in pos_tokens}
#         token_sentiment.update({token: -1 for token in neg_tokens})
#
#         self.pos_tokens = pos_tokens
#         self.neg_tokens = neg_tokens
#         self.token_sentiment = token_sentiment
#
#     def tokens_to_sentiments(self, tokens):
#         """Return sentiments of tokens in a sentence
#         """
#         sentiments = np.array([self.token_sentiment.get(token, 0) for token in tokens])
#
#         return sentiments
#
#     def tokens_with_sentiment(self, tokens, sentiment):
#         """Return tokens with specified sentiment
#         """
#         sentiments = self.tokens_to_sentiments(tokens)
#         tokens = np.array(tokens)[sentiments == sentiment]
#
#         return tokens


