import numpy as np
import optuna.logging
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from sklearn.metrics import f1_score, accuracy_score

optuna.logging.disable_default_handler()


def get_label_model(method, cardinality=2):
    if method == "mv":
        return MajorityLabelVoter(cardinality=cardinality)
    elif method == "snorkel":
        snorkel_kwargs = {
            'search_space': {
                # 'optimizer': ['sgd'],
                'lr': np.logspace(-4, -1, num=4, base=10),
                'l2': np.logspace(-4, -1, num=4, base=10),
                'n_epochs': [5, 10, 50, 100],
            },
            'cardinality': cardinality,
            'n_trials': 512
        }
        return Snorkel(**snorkel_kwargs)


class Snorkel:
    def __init__(self, seed=None, **kwargs):
        self.search_space = kwargs['search_space']
        self.n_trials = kwargs.get('n_trials', 100)
        self.cardinality = kwargs['cardinality']
        self.best_params = None
        self.model = None
        self.kwargs = kwargs
        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed

    def _to_onehot(self, ys):
        ys = np.array(ys)
        ys[ys == -1] = 0
        ys_onehot = np.zeros((len(ys), self.cardinality))
        ys_onehot[range(len(ys_onehot)), ys] = 1

        return ys_onehot

    def fit(self, L_tr, L_val, ys_val, scoring='logloss', tune_label_model=True):
        search_space = self.search_space

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            nonlocal L_tr, L_val, ys_val
            model = LabelModel(cardinality=self.cardinality, verbose=False)
            model.fit(L_train=L_tr, Y_dev=ys_val, **suggestions, seed=self.seed, progress_bar=False)

            # logloss as validation loss
            if scoring == 'logloss':
                ys_pred_val = model.predict_proba(L_val)
                ys_val_onehot = self._to_onehot(ys_val)
                val_loss = -(ys_val_onehot * np.log(np.clip(ys_pred_val, 1e-6, 1.))).sum(axis=1).mean()
            elif scoring == 'f1':
                ys_pred = model.predict(L_val)
                active_indices = ys_pred != -1
                if self.cardinality > 2:
                    val_loss = -f1_score(ys_val[active_indices], ys_pred[active_indices], average="macro")
                else:
                    val_loss = -f1_score(ys_val[active_indices], ys_pred[active_indices])
            elif scoring == "acc":
                ys_pred = model.predict(L_val)
                active_indices = ys_pred != -1
                val_loss = -accuracy_score(ys_val[active_indices], ys_pred[active_indices])
            else:
                raise ValueError("Scoring metric not supported.")

            return val_loss

        # search for best hyperparameter
        if tune_label_model:
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
            study.optimize(objective, n_trials=self.n_trials)
            self.best_params = study.best_params
        else:
            self.best_params = {}
        self.model = LabelModel(cardinality=self.cardinality,verbose=False)
        self.model.fit(L_train=L_tr, Y_dev=ys_val, **self.best_params, seed=self.seed, progress_bar=False)

    def predict_proba(self, L):
        return self.model.predict_proba(L)

    def predict(self, L):
        return self.model.predict(L, tie_break_policy="abstain")
