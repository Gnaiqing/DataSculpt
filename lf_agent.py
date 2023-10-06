import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from numpy.random import default_rng
from lf_family import KeywordLF, RegexLF
import openai
from gpt_utils import create_prompt, create_cot_prompt, create_cot_user_prompt, create_user_prompt, extract_response, build_example
from sentence_transformers import SentenceTransformer
from data_utils import preprocess_text
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
        if self.example_selection == "neighbor":
            # compute the cosine similarity between training instances (unlabeled) and validation instances (labeled)
            embedding_model = kwargs.get("embedding_model", "all-MiniLM-L12-v2")
            self.embedding_model = SentenceTransformer(embedding_model)
            train_sentences = [self.train_dataset.examples[i]["text"] for i in range(len(self.train_dataset))]
            valid_sentences = [self.valid_dataset.examples[i]["text"] for i in range(len(self.valid_dataset))]
            train_embedding = self.embedding_model.encode(train_sentences)
            valid_embedding = self.embedding_model.encode(valid_sentences)
            self.similarity_matrix = cosine_similarity(train_embedding, valid_embedding)
            self.neighbors = np.argsort(- self.similarity_matrix, axis=-1)

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
                                                                lf_type=self.lf_type)
        self.cot_system_prompt, self.cot_example_prompt = create_cot_prompt(self.kwargs["dataset_name"], self.valid_dataset,
                                                                            example_per_class=self.example_per_class,
                                                                            example_selection=self.example_selection,
                                                                            explanation=self.return_explanation,
                                                                            lf_type=self.lf_type)
        if self.display:
            print("ChatGPT system prompt:")
            print(self.system_prompt)
            print("Example prompt:")
            print(self.example_prompt)

    def create_lf(self, query_idx):
        if self.example_selection == "neighbor":
            # select the examples that are closest to the test instance
            n_examples = self.example_per_class * self.train_dataset.n_class
            cur_examples = 0
            k = 0
            example_string = ""
            while cur_examples < n_examples:
                valid_idx = self.neighbors[query_idx][k]
                k += 1
                cot_user_prompt = create_cot_user_prompt(self.cot_example_prompt, self.kwargs["dataset_name"], self.valid_dataset, valid_idx)
                messages = [
                    {"role": "system", "content": self.cot_system_prompt},
                    {"role": "user", "content": cot_user_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        n=self.n_completion
                    )
                    response_content = response['choices'][0]["message"]["content"]
                except openai.error.OpenAIError:
                    response_content = ""

                response_dict = extract_response(response_content)
                if self.lf_type == "keyword" and response_dict["keyword_list"] is not None and len(response_dict["keyword_list"]) > 0:
                    cur_examples += 1
                    example_string += build_example(self.kwargs["dataset_name"], self.valid_dataset, valid_idx, response_dict)
                elif self.lf_type == "regex" and response_dict["regex_list"] is not None and len(response_dict["regex_list"]) > 0:
                    cur_examples += 1
                    example_string += build_example(self.kwargs["dataset_name"], self.valid_dataset, valid_idx,
                                                    response_dict)
            if self.display:
                print("In context examples:")
                print(example_string)

        else:
            example_string = self.example_prompt

        user_prompt = create_user_prompt(example_string, self.kwargs["dataset_name"], self.train_dataset, query_idx)
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

                    response_dict = extract_response(response_content)
                    label = response_dict["label"]
                    keywords = response_dict["keyword_list"]
                    # label, keywords = extract_label_keywords(response_content, self.train_dataset.n_class)
                    if label in range(self.train_dataset.n_class):
                        output_labels.append(label)
                    if isinstance(keyword_list, list):
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

        elif self.lf_type == "regex":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            label, regex_list = None, []
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
                    response_dict = extract_response(response_content)
                    label = response_dict["label"]
                    regexs = response_dict["regex_list"]
                    # label, regexs = extract_label_regex(response_content, self.train_dataset.n_class)
                    if label in range(self.train_dataset.n_class):
                        output_labels.append(label)
                    if isinstance(regexs, list):
                        regex_list += regexs

                if len(output_labels) > 0:
                    label = np.bincount(output_labels).argmax()
                    break

            regex_list = np.unique(regex_list)
            if label is not None:
                for regex in regex_list:
                    try:
                        # check if regex is a valid regular expression
                        re.compile(regex)
                    except re.error:
                        continue
                    lf = RegexLF(regex=regex, label=label)
                    candidate_lfs.append(lf)

        else:
            raise NotImplementedError("LF Type not supported.")

        filtered_lfs = filter_candidate_lfs(candidate_lfs, self.filter_methods, self.train_dataset,
                                            self.valid_dataset, self.acc_threshold, self.lfs)

        if self.max_lf_per_iter is not None and self.max_lf_per_iter < len(filtered_lfs):
            filtered_lfs = self.rng.choice(filtered_lfs, size=self.max_lf_per_iter, replace=False)

        if len(filtered_lfs) == 0:
            if self.lf_type == "keyword":
                filtered_lfs = [KeywordLF("none", label)]
            else:
                filtered_lfs = [RegexLF("none", label)]
        else:
            for lf in filtered_lfs:
                self.lfs.append(lf)

        return filtered_lfs


class SimLFAgent:
    def __init__(self, train_dataset, valid_dataset, lf_type="keyword", filter_methods=("acc", "unique"),
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
            raise ValueError("LF Type not supported.")

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
