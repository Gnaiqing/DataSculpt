import numpy as np
import nltk
from sklearn.metrics import accuracy_score
class KeywordLF:
    def __init__(self, keyword, label):
        self.keyword = keyword
        self.label = label

    def __eq__(self, other):
        if isinstance(other, KeywordLF):
            if self.keyword == other.keyword and self.label == other.label:
                return True

        return False

    def apply(self, x):
        tokens = nltk.word_tokenize(x)
        if self.keyword in tokens:
            return self.label
        else:
            return -1

    def apply_to_dataset(self, dataset):
        wl = np.repeat(-1, len(dataset))
        if self.keyword in dataset.revert_index:
            active_indices = dataset.revert_index[self.keyword]
            wl[active_indices] = self.label

        return wl

    def get_active_indices(self, dataset):
        if self.keyword in dataset.revert_index:
            active_indices = dataset.revert_index[self.keyword]
        else:
            active_indices = np.array([])

        return active_indices

    def get_cov_acc(self, dataset):
        active_indices = self.get_active_indices(dataset)
        if len(active_indices) == 0:
            return 0.0, np.nan

        ys = np.array(dataset.labels)[active_indices]
        if -1 in ys:
            # the dataset has missing ground-truth labels
            acc = np.nan
        else:
            acc = np.mean(ys == self.label)

        cov = len(active_indices) / len(dataset)
        return cov, acc

    def info(self):
        return f"{self.keyword}->{self.label}"


def create_label_matrix(dataset, lfs):
    weak_labels = []
    for lf in lfs:
        wl = lf.apply_to_dataset(dataset)
        weak_labels.append(wl.reshape(-1,1))

    weak_labels = np.hstack(weak_labels)
    return weak_labels


def check_all_class(ys, cardinality):
    mask = np.repeat(False, cardinality)
    for label in ys:
        mask[label] = True

    return np.all(mask)