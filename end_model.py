import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
import pdb


def get_discriminator(model_type, prob_labels, params=None, seed=None):
    if model_type == 'logistic':
        return LogReg(prob_labels, params, seed)
    else:
        raise ValueError('discriminator model not supported.')

def to_onehot(ys, cardinality=2):
    ys_onehot = np.zeros((len(ys),cardinality), dtype=float)
    ys_onehot[range(len(ys_onehot)), ys] = 1.0
    return ys_onehot

def evaluate_disc_model(disc_model, test_dataset):
    y_pred = disc_model.predict(test_dataset.xs_feature)
    y_probs = disc_model.predict_proba(test_dataset.xs_feature)
    test_acc = accuracy_score(test_dataset.ys, y_pred)
    if test_dataset.n_class == 2:
        test_auc = roc_auc_score(test_dataset.ys, y_probs[:, 1])
        test_f1 = f1_score(test_dataset.ys, y_pred)
    else:
        test_auc = roc_auc_score(test_dataset.ys, y_probs, average="macro", multi_class="ovo")
        test_f1 = f1_score(test_dataset.ys, y_pred, average="macro")

    results = {
        "acc": test_acc,
        "auc": test_auc,
        "f1": test_f1
    }
    return results

def train_disc_model(model_type, xs_tr, ys_tr_soft, ys_tr_hard, valid_dataset, warmup_dataset, soft_training, seed):
    # prepare discriminator
    disc_model = get_discriminator(model_type=model_type, prob_labels=soft_training, seed=seed)

    if soft_training:
        # ys_tr = ys_tr_soft[:, 1]
        # ys_warmup = (warmup_dataset.ys == 1).astype(float)
        ys_tr = ys_tr_soft
        ys_warmup = to_onehot(warmup_dataset.ys, warmup_dataset.n_class)
    else:
        ys_tr = ys_tr_hard
        ys_warmup = warmup_dataset.ys

    if xs_tr is None:
        assert len(warmup_dataset) > 0
        xs_tr = warmup_dataset.xs_feature
        ys_tr = ys_warmup
    else:
        xs_tr = np.vstack((xs_tr, warmup_dataset.xs_feature))
        ys_tr = np.hstack((ys_tr, ys_warmup))

    sample_weights = None
    disc_model.tune_params(xs_tr, ys_tr, valid_dataset.xs_feature, valid_dataset.ys, sample_weights)
    disc_model.fit(xs_tr, ys_tr, sample_weights)

    return disc_model


class Classifier:
    """Classifier backbone
    """

    def tune_params(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    def fit(self, xs, ys):
        raise NotImplementedError

    def predict(self, xs):
        raise NotImplementedError


class LogReg(Classifier):
    def __init__(self, prob_labels, params=None, seed=None):
        self.prob_labels = prob_labels
        self.model = None
        self.best_params = None
        if params is None:
            params = {
                'solver': ['liblinear'],
                'max_iter': [1000],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
            }
        self.params = params
        self.n_trials = 10
        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed

    def tune_params(self, x_train, y_train, x_valid, y_valid, sample_weights=None, scoring='acc'):
        search_space = self.params

        if self.prob_labels:
            # use weighted data to simulate prob labels
            cardinality = y_train.shape[1]
            x_train = np.vstack([x_train] * cardinality)
            weights = y_train.T.reshape(-1)
            y_train_hard = []
            for c in range(cardinality):
                y_train_hard.append(np.ones(len(y_train))*c)

            y_train = np.hstack(y_train_hard)

            if sample_weights is not None:
                sample_weights = np.hstack([sample_weights] * cardinality) * weights
            else:
                sample_weights = weights

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}

            model = LogisticRegression(**suggestions, random_state=self.seed)
            model.fit(x_train, y_train, sample_weights)

            ys_pred = model.predict(x_valid)

            if scoring == 'acc':
                val_score = accuracy_score(y_valid, ys_pred)
            elif scoring == 'f1':
                val_score = f1_score(y_valid, ys_pred)

            return val_score

        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params

    def fit(self, xs, ys, sample_weights=None):
        if self.prob_labels:
            cardinality = ys.shape[1]
            xs = np.vstack([xs] * cardinality)
            weights = ys.T.reshape(-1)
            y_train_hard = []
            for c in range(cardinality):
                y_train_hard.append(np.ones(len(ys)) * c)

            ys = np.hstack(y_train_hard)

            if sample_weights is not None:
                sample_weights = np.hstack([sample_weights] * cardinality) * weights
            else:
                sample_weights = weights

        if self.best_params is not None:
            model = LogisticRegression(**self.best_params, random_state=self.seed)
            model.fit(xs, ys, sample_weight=sample_weights)
            self.model = model
        else:
            raise ValueError('Should perform hyperparameter tuning before fitting')

    def predict(self, xs):
        return self.model.predict(xs)

    def predict_proba(self, xs):
        return self.model.predict_proba(xs)