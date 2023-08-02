import numpy as np
from data_utils import TextDataset, TextPairDataset
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
        if self.keyword in x["token"]:
            return self.label
        else:
            return -1

    def apply_to_dataset(self, dataset:TextDataset):
        wl = np.repeat(-1, len(dataset))
        if self.keyword in dataset.revert_index:
            active_indices = dataset.revert_index[self.keyword]
            wl[active_indices] = self.label

        return wl
    def get_active_indices(self, dataset:TextDataset):
        if self.keyword in dataset.revert_index:
            active_indices = dataset.revert_index[self.keyword]
        else:
            active_indices = np.array([])

        return active_indices

    def get_cov_acc(self, dataset:TextDataset):
        active_indices = self.get_active_indices(dataset)
        ys = dataset.ys[active_indices]
        if -1 in ys:
            # the dataset has missing ground-truth labels
            acc = np.nan
        else:
            acc = np.mean(ys == self.label)

        cov = len(active_indices) / len(dataset)
        return cov, acc

    def info(self):
        return f"{self.keyword}->{self.label}"


def apply_lfs(lfs, dataset):
    weak_labels = []
    for lf in lfs:
        wl = lf.apply_to_dataset(dataset)
        weak_labels.append(wl.reshape(-1,1))

    weak_labels = np.hstack(weak_labels)
    return weak_labels


def check_all_class(lfs, cardinality):
    mask = np.repeat(False, cardinality)
    for lf in lfs:
        mask[lf.label] = True

    return np.all(mask)