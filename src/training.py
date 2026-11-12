from .utils import set_seed, resolve_path
from sklearn.base import clone
from .data import Dataset
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error
from sklearn.exceptions import NotFittedError
from joblib import dump, load
import polars as pl
import numpy as np


class Training:
    def __init__(self, model, cv, dataset, seed):
        self.model = model
        self.cv = cv
        self.dataset = dataset
        self.seed = seed
        self.fitted_model = None

    def fit_single_sklearn_model(self, refit=False):
        set_seed(self.seed)
        model = clone(self.model)

        results = {"cv_loss": [], "avg_loss": None}
        X, y = self.dataset.X, self.dataset.y
        split_loop = tqdm(self.cv.split(X, y), total=self.cv.get_n_splits(), desc="CV")
        cv_loss = 0
        for i, (train_idx, val_idx) in enumerate(split_loop, start=1):
            fold_model = clone(model)
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]
            fold_model.fit(X_train_fold, y_train_fold)
            fold_pred = fold_model.predict(X_val_fold)
            fold_rmse = root_mean_squared_error(y_val_fold, fold_pred)
            cv_loss += fold_rmse
            results["cv_loss"].append(fold_rmse)
        cv_loss /= self.cv.get_n_splits()
        results["avg_loss"] = cv_loss
        if refit:
            self.fitted_model = model.fit(X, y)
        return results

    def get_predicts(
        self,
        test_path,
        model_path=None,
    ):
        if self.fitted_model is None and model_path is None:
            raise NotFittedError(
                "No trained model: call fit_single_sklearn_model(refit=True) first or pass model_path"
            )
        if model_path is not None:
            model_path = resolve_path(model_path)
            model = load(model_path)
        else:
            model = self.fitted_model

        ids, X_test = self.dataset.load_test(test_path)
        pred = model.predict(X_test)
        pred = np.expm1(pred)
        submission = ids.with_columns(pl.Series(pred).alias("SalePrice"))
        return submission

    def save_model(self, path=None):
        if self.fitted_model is None:
            raise NotFittedError(
                "No trained model: call fit_single_sklearn_model() first"
            )

        path = resolve_path("artefacts/models") if path is None else resolve_path(path)
        path.mkdir(parents=True, exist_ok=True)
        dump(self.fitted_model, path)
        return path
