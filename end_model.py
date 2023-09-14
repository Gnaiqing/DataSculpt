import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
import optuna
import pdb


def get_discriminator(model_type, prob_labels, params=None, seed=None, ssl_method=None, n_class=2):
    if model_type == 'logistic':
        if ssl_method is None:
            return LogReg(prob_labels, params, seed, n_class=n_class)
        else:
            return LogRegSSL(params, seed, n_class=n_class)
    else:
        raise ValueError('discriminator model not supported.')


def to_onehot(ys, cardinality=2):
    ys_onehot = np.zeros((len(ys),cardinality), dtype=float)
    ys_onehot[range(len(ys_onehot)), ys] = 1.0
    return ys_onehot


def train_disc_model(model_type, xs_tr, ys_tr_soft, ys_tr_hard, xs_u, valid_dataset, soft_training,
                     ssl_method, tune_end_model, tune_metric, seed):
    # prepare discriminator
    disc_model = get_discriminator(model_type=model_type, prob_labels=soft_training, seed=seed, ssl_method=ssl_method,
                                   n_class=valid_dataset.n_class)

    if soft_training:
        ys_tr = ys_tr_soft
    else:
        ys_tr = ys_tr_hard

    if ssl_method is None:
        sample_weights = None
        if tune_end_model:
            disc_model.tune_params(xs_tr, ys_tr, valid_dataset.features, valid_dataset.labels, sample_weights, scoring=tune_metric)
        else:
            disc_model.best_params = {}
        disc_model.fit(xs_tr, ys_tr, sample_weights)
    else:
        if tune_end_model:
            disc_model.tune_params(xs_tr, ys_tr, valid_dataset.features, valid_dataset.labels, xs_u, scoring=tune_metric)
        else:
            disc_model.best_params = {}
        disc_model.fit(xs_tr, ys_tr, xs_u)

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
    def __init__(self, prob_labels, params=None, seed=None, n_class=2):
        self.prob_labels = prob_labels
        self.model = None
        self.best_params = None
        self.n_class = n_class
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

    def _to_onehot(self, ys):
        ys = np.array(ys)
        ys[ys == -1] = 0
        ys_onehot = np.zeros((len(ys), self.n_class))
        ys_onehot[range(len(ys_onehot)), ys] = 1
        return ys_onehot

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

            # remove data instance with zero weights
            active_idxs = sample_weights != 0
            x_train = x_train[active_idxs, :]
            y_train = y_train[active_idxs]
            sample_weights = sample_weights[active_idxs]

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            model = LogisticRegression(**suggestions, random_state=self.seed)
            model.fit(x_train, y_train, sample_weights)
            ys_pred = model.predict(x_valid)
            if scoring == 'logloss':
                ys_pred_val = model.predict_proba(x_valid)
                ys_val_onehot = self._to_onehot(y_valid)
                val_score = (ys_val_onehot * np.log(np.clip(ys_pred_val, 1e-6, 1.))).sum(axis=1).mean()
            if scoring == 'acc':
                val_score = accuracy_score(y_valid, ys_pred)
            elif scoring == 'f1':
                if self.n_class > 2:
                    val_score = f1_score(y_valid, ys_pred, average="macro")
                else:
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

            # remove data instance with zero weights
            active_idxs = sample_weights != 0
            xs = xs[active_idxs, :]
            ys = ys[active_idxs]
            sample_weights = sample_weights[active_idxs]

        self.active_classes = np.unique(ys).astype(int)  # classes that the model returns
        if self.best_params is not None:
            model = LogisticRegression(**self.best_params, random_state=self.seed)
            model.fit(xs, ys, sample_weight=sample_weights)
            self.model = model
        else:
            raise ValueError('Should perform hyperparameter tuning before fitting')

    def predict(self, xs):
        return self.model.predict(xs)

    def predict_proba(self, xs):
        proba = np.zeros((len(xs), self.n_class))
        proba[:, self.active_classes] = self.model.predict_proba(xs)
        return proba
        # return self.model.predict_proba(xs)


class LogRegSSL(Classifier):
    """
    Apply self training for semi-supervised learning
    """
    def __init__(self, params=None, seed=None, n_class=2):
        self.model = None
        self.best_params = None
        self.n_class = n_class
        if params is None:
            params = {
                'LogReg': {
                    'solver': ['liblinear'],
                    'max_iter': [1000],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                },
                'SelfTraining': {
                    'threshold': [0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
                }
            }
        self.params = params
        self.n_trials = 50
        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed

    def tune_params(self, x_train, y_train, x_valid, y_valid, x_unlabeled, scoring='acc'):
        search_space = {}
        for key in self.params:
            search_space = {**search_space, **self.params[key]}

        x_train = np.vstack((x_train, x_unlabeled))
        y_train = np.hstack((y_train, - np.ones(len(x_unlabeled))))
        x_train, y_train = shuffle(x_train, y_train)

        def objective(trial):
            logreg_suggestions = {key: trial.suggest_categorical(key, self.params['LogReg'][key]) for key in self.params['LogReg']}
            base_model = LogisticRegression(**logreg_suggestions, random_state=self.seed)
            ssl_suggestions = {key: trial.suggest_categorical(key, self.params['SelfTraining'][key]) for key in self.params['SelfTraining']}
            model = SelfTrainingClassifier(base_model, **ssl_suggestions)
            model.fit(x_train, y_train)
            ys_pred = model.predict(x_valid)

            if scoring == 'acc':
                val_score = accuracy_score(y_valid, ys_pred)
            elif scoring == 'f1':
                val_score = f1_score(y_valid, ys_pred)

            return val_score

        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params

    def fit(self, xs, ys, xu):
        x_train = np.vstack((xs, xu))
        y_train = np.hstack((ys, - np.ones(len(xu))))
        x_train, y_train = shuffle(x_train, y_train)
        logreg_params = {key : self.best_params[key] for key in ['solver', 'max_iter', 'C']}
        self_training_params = {key: self.best_params[key] for key in ['threshold']}
        if self.best_params is not None:
            base_model = LogisticRegression(**logreg_params, random_state=self.seed)
            model = SelfTrainingClassifier(base_model, **self_training_params)

            model.fit(x_train, y_train)
            self.model = model
        else:
            raise ValueError('Should perform hyperparameter tuning before fitting')

    def predict(self, xs):
        return self.model.predict(xs)

    def predict_proba(self, xs):
        return self.model.predict_proba(xs)