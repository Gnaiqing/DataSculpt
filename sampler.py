import abc
import numpy as np
import random
from alipy import ToolBox
from sentence_transformers import SentenceTransformer, util
from scipy.stats import entropy
from data_utils import build_revert_index


def get_sampler(train_dataset, sampler_type, **kwargs):
    if sampler_type in ["passive", "uncertain", "QBC"]:
        return ActiveSampler(train_dataset, al_method=sampler_type, **kwargs)
    elif sampler_type == "SEU":
        return SEUSampler(train_dataset)
    elif sampler_type == "weighted":
        return WeightedScoreSampler(train_dataset, **kwargs)
    else:
        raise ValueError("Sampler not supported")


class Sampler(abc.ABC):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

    def sample(self, batch_size=1, **kwargs):
        pass


class ActiveSampler(Sampler):
    """
    Sampler that leverages off-the-box active learning methods
    """
    def __init__(self, dataset, al_method="uncertain", seed=0, **kwargs):
        super(ActiveSampler, self).__init__(dataset)
        self.al_method = al_method
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.alibox = ToolBox(X=dataset.features, y=dataset.labels, query_type="AllLabels")
        self.label_index = self.alibox.IndexCollection([0]).difference_update([0])
        self.unlabel_index = self.alibox.IndexCollection(np.arange(len(dataset)))

    def sample(self, batch_size=1, lm_probs=None, al_model=None):
        if al_model is None or self.al_method == "passive":
            strategy = self.alibox.get_query_strategy(strategy_name="QueryInstanceRandom")
        elif self.al_method == "uncertain":
            strategy = self.alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty")
        elif self.al_method == "QBC":
            strategy = self.alibox.get_query_strategy(strategy_name="QueryInstanceQBC")
        else:
            raise ValueError("AL method not supported.")

        indices = strategy.select(label_index=self.label_index,
                                  unlabel_index=self.unlabel_index,
                                  model=al_model,
                                  batch_size=batch_size)

        self.label_index.update(indices)
        self.unlabel_index.difference_update(indices)
        return indices


class SEUSampler(Sampler):
    """
    Select by expected utility (Nemo)
    """
    def __init__(self, dataset, **kwargs):
        super(SEUSampler, self).__init__(dataset)
        if hasattr(dataset, "revert_index"):
            self.revert_index = dataset.revert_index
        else:
            self.revert_index = build_revert_index(dataset)

        self.active_keyword_index = [[] for _ in range(len(dataset))]
        for (keyword_index, keyword) in enumerate(self.revert_index):
            active_index = self.revert_index[keyword]
            for i in active_index:
                self.active_keyword_index[i].append(keyword_index)

        self.n_class = dataset.n_class
        self.label_index = np.array([], dtype=int)
        self.unlabel_index = np.arange(len(dataset), dtype=int)

    def sample(self, batch_size=1, lm_probs=None):
        if lm_probs is None:
            sampled_index = np.random.choice(self.unlabel_index, size=batch_size, replace=False)
        else:
            y_preds = np.argmax(lm_probs, axis=1)
            lf_accs = np.zeros((len(self.revert_index), self.n_class), dtype=float)
            # estimate LF accuracy
            for (keyword_index, keyword) in enumerate(self.revert_index):
                for c in range(self.n_class):
                    lf_acc = np.mean(y_preds[self.revert_index[keyword]] == c)
                    lf_accs[keyword_index, c] = lf_acc

            # estimate LF utility
            lf_scores = np.zeros_like(lf_accs, dtype=float)
            uncertain_score = entropy(lm_probs, axis=1)
            for (keyword_index, keyword) in enumerate(self.revert_index):
                for c in range(self.n_class):
                    active_indices = self.revert_index[keyword]
                    pos_indices = active_indices[y_preds[active_indices] == c]
                    neg_indices = active_indices[y_preds[active_indices] != c]
                    lf_scores[keyword_index, c] = np.sum(uncertain_score[pos_indices]) - np.sum(uncertain_score[neg_indices])

            # compute score per instance
            unlabel_scores = []
            for idx in self.unlabel_index:
                lf_acc_list = []
                lf_score_list = []
                for keyword_index in self.active_keyword_index[idx]:
                    for c in range(self.n_class):
                        lf_acc_list.append(lf_accs[keyword_index, c])
                        lf_score_list.append(lf_scores[keyword_index, c])

                lf_probs = np.array(lf_acc_list) / np.sum(lf_acc_list)
                score = np.sum(np.array(lf_score_list) * lf_probs)
                unlabel_scores.append(score)

            unlabel_pos = np.argsort(unlabel_scores)[-1:-batch_size-1:-1]
            sampled_index = self.unlabel_index[unlabel_pos]

        self.label_index = np.union1d(self.label_index, sampled_index)
        self.unlabel_index = np.setdiff1d(self.unlabel_index, sampled_index)
        return sampled_index


class WeightedScoreSampler(Sampler):
    def __init__(self, dataset, embedding_model="all-MiniLM-L12-v2", distance="cosine",
                 uncertain_metric="entropy", k=100, alpha=0.5, beta=0.0, **kwargs):
        super(WeightedScoreSampler, self).__init__(dataset)
        self.embedding_model = SentenceTransformer(embedding_model)
        sentences = [dataset.examples[i]["text"] for i in range(len(dataset))]
        self.embeddings = self.embedding_model.encode(sentences)
        if distance == "cosine":
            self.sim_func = util.cos_sim
        else:
            raise ValueError("Similarity function not supported.")

        self.dist_mat = 1 - self.sim_func(self.embeddings, self.embeddings).numpy()
        self.n = len(self.dataset)
        self.k = k  # neighbor count
        self.alpha = alpha  # trade-off factor for uncertain score and distance
        self.beta = beta  # decay factor for distance metric
        self.uncertain_metric = uncertain_metric
        # compute k-nearest neighbors for each point
        self.neighbors = np.argsort(self.dist_mat, axis=1)[:,1:k+1]
        self.neighbor_dists = np.sort(self.dist_mat, axis=1)[:,1:k+1]
        self.label_index = np.array([], dtype=int)
        self.unlabel_index = np.arange(len(dataset), dtype=int)

    def distance_to_labeled(self, indices):
        if len(self.label_index) == 0:
            return np.repeat(100, len(indices))
        distance = self.dist_mat[np.ix_(indices, self.label_index)]
        sorted_distance = np.sort(distance, axis=1)
        dist = sorted_distance[:,0]
        dist[dist < 0.0] = 0.0  # avoid some float computation issue
        return dist

    def sample(self, batch_size=1, lm_probs=None):
        if lm_probs is None:
            uncertain_score = np.ones(self.n)
        else:
            if self.uncertain_metric == "entropy":
                uncertain_score = entropy(lm_probs, axis=1)
            elif self.uncertain_metric == "confidence":
                uncertain_score = 1 - np.max(lm_probs, axis=1)
            elif self.uncertain_metric == "margin":
                sorted_probs = np.sort(lm_probs, axis=1)
                uncertain_score = 1 - (sorted_probs[:,-1] - sorted_probs[:,-2])
            else:
                raise ValueError("Uncertainty score not supported.")
            # # normalize the uncertain score to [0,1]
            # min_score = np.min(uncertain_score)
            # max_score = np.max(uncertain_score)
            # if min_score != max_score:
            #     uncertain_score = (uncertain_score - min_score) / (max_score - min_score)
            # # transform uncertain score to be positive
            # uncertain_score = np.exp(uncertain_score)

        weights = np.power(self.distance_to_labeled(np.arange(self.n)), self.alpha) * \
                  np.power(uncertain_score, 1-self.alpha)
        neighbor_weights = weights[self.neighbors]
        scores = np.sum(neighbor_weights * np.exp(- self.neighbor_dists * self.beta), axis=1)
        unlabel_scores = scores[self.unlabel_index]
        unlabel_pos = np.argsort(unlabel_scores)[-1:-batch_size-1:-1]
        sampled_index = self.unlabel_index[unlabel_pos]
        self.label_index = np.union1d(self.label_index, sampled_index)
        self.unlabel_index = np.setdiff1d(self.unlabel_index, sampled_index)
        return sampled_index





