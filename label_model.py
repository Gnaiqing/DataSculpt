import numpy as np
import optuna.logging
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import warnings

optuna.logging.disable_default_handler()


def get_label_model(method, cardinality=2, **kwargs):
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
        return Snorkel(**snorkel_kwargs, **kwargs)


class Snorkel:
    def __init__(self, seed=None, calibration="isotonic", **kwargs):
        self.search_space = kwargs['search_space']
        self.n_trials = kwargs.get('n_trials', 100)
        self.cardinality = kwargs['cardinality']
        self.calibration = calibration
        if self.calibration is not None and self.cardinality > 2:
            warnings.warn("Currently, calibration is only supported for binary classification.")
            self.calibration = None

        self.best_params = None
        self.model = None
        self.kwargs = kwargs
        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed
        if self.calibration == "sigmoid":
            self.calib_model = LogisticRegression()
            self.threshold = 0.5
        elif self.calibration == "isotonic":
            self.calib_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            self.threshold = 0.5
        else:
            self.calib_model = None

    def _to_onehot(self, ys):
        ys = np.array(ys)
        ys[ys == -1] = 0
        ys_onehot = np.zeros((len(ys), self.cardinality))
        ys_onehot[range(len(ys_onehot)), ys] = 1

        return ys_onehot

    def fit(self, L_tr, L_val, ys_val, scoring='logloss', tune_label_model=True):
        search_space = self.search_space
        train_covered_indices = np.max(L_tr, axis=1) != -1
        valid_covered_indices = np.max(L_val, axis=1) != -1
        L_tr_filtered = L_tr[train_covered_indices, :]
        L_val_filtered = L_val[valid_covered_indices, :]
        ys_val_filtered = ys_val[valid_covered_indices]

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            nonlocal L_tr, L_val, ys_val
            model = LabelModel(cardinality=self.cardinality, verbose=False)
            model.fit(L_train=L_tr, Y_dev=ys_val, **suggestions, seed=self.seed, progress_bar=False)

            if len(L_val_filtered) == 0:
                return 0.0  # no data in validation set is covered

            # logloss as validation loss
            if scoring == 'logloss':
                ys_pred_val = model.predict_proba(L_val_filtered)
                ys_val_onehot = self._to_onehot(ys_val_filtered)
                val_loss = -(ys_val_onehot * np.log(np.clip(ys_pred_val, 1e-6, 1.))).sum(axis=1).mean()
            elif scoring == 'f1':
                ys_pred = model.predict(L_val_filtered)
                if self.cardinality > 2:
                    val_loss = -f1_score(ys_val_filtered, ys_pred, average="macro")
                else:
                    val_loss = -f1_score(ys_val_filtered, ys_pred)
            elif scoring == "acc":
                ys_pred = model.predict(L_val_filtered)
                val_loss = -accuracy_score(ys_val_filtered, ys_pred)
            else:
                raise ValueError("Scoring metric not supported.")

            return val_loss

        # search for best hyperparameter
        if tune_label_model and len(ys_val_filtered) > 0:
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
            study.optimize(objective, n_trials=self.n_trials)
            self.best_params = study.best_params
        else:
            self.best_params = {}
        self.model = LabelModel(cardinality=self.cardinality, verbose=False)
        self.model.fit(L_train=L_tr, Y_dev=ys_val, **self.best_params, seed=self.seed, progress_bar=False)
        if self.calibration is not None:
            # currently, only support calibration for binary classification
            X_val_filtered = self.model.predict_proba(L_val_filtered)[:,1]  # uncalibrated probability
            self.calib_model.fit(X_val_filtered, ys_val_filtered)  # calibrate probability prediction
            # use validation set to select decision threshold
            X_val_filtered = self.model.predict_proba(L_val_filtered)[:, 1]
            if self.calibration == "sigmoid":
                p_val_filtered = self.calib_model.predict_proba(X_val_filtered)[:,1]
            elif self.calibration == "isotonic":
                p_val_filtered = self.calib_model.predict(X_val_filtered)

            candidate_thres = np.unique(p_val_filtered)
            best_score = 0.0
            if len(ys_val_filtered) > 0:
                for theta in candidate_thres:
                    y_pred = p_val_filtered >= theta
                    if scoring == "f1":
                        score = f1_score(ys_val_filtered, y_pred)
                    else:
                        score = accuracy_score(ys_val_filtered, y_pred)

                    if score > best_score:
                        best_score = score
                        self.threshold = theta

    def predict_proba(self, L):
        proba = self.model.predict_proba(L)
        if self.calibration == "sigmoid":
            proba = self.calib_model.predict_proba(proba[:,1])
        elif self.calibration == "isotonic":
            p = self.calib_model.predict(proba[:,1])
            proba[:,1] = p
            proba[:,0] = 1-p

        return proba

    def predict(self, L):
        proba = self.predict_proba(L)
        if self.calibration is not None:
            pred = (proba[:,1] >= self.threshold).astype(int)
            return pred
        else:
            return np.argmax(proba, axis=1)

        # return self.model.predict(L, tie_break_policy="abstain")
