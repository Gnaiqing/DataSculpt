import abc
import numpy as np
import random
from alipy import ToolBox


def get_sampler(train_dataset, sampler_type, **kwargs):
    if sampler_type in ["passive", "uncertain", "QBC"]:
        return ActiveSampler(train_dataset, sampler_type, **kwargs)
    else:
        raise ValueError("Sampler not supported")


class Sampler(abc.ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    def sample(self):
        pass


class ActiveSampler(Sampler):
    def __init__(self, dataset, al_method, seed):
        super(ActiveSampler, self).__init__(dataset)
        self.al_method = al_method
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.alibox = ToolBox(X=dataset.xs_feature, y=dataset.ys, query_type="AllLabels")
        self.label_index = self.alibox.IndexCollection([0]).difference_update([0])
        self.unlabel_index = self.alibox.IndexCollection(np.arange(len(dataset)))

    def sample(self, al_model=None, batch_size=1):
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



