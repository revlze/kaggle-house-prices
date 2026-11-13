import numpy as np
import polars as pl
from joblib import dump, load
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

from .utils import resolve_path, set_seed


class Training:
    """Cross-validate an sklearn model on a Dataset and predict the test set."""

    def __init__(self, model, cv, dataset, seed):
        self.model = model
        self.cv = cv
        self.dataset = dataset
        self.seed = seed
        self.fitted_model = None
        self.fold_models = []

    def fit_single_sklearn_model(self, refit=False):
        """Run CV and return {"cv_loss": per-fold RMSE, "avg_loss": mean RMSE}.

        Fold models are kept in self.fold_models; with refit=True the model is also
        trained on all rows and stored in self.fitted_model.
        """
        set_seed(self.seed)
        self.fold_models = []
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
            self.fold_models.append(fold_model)
            fold_pred = fold_model.predict(X_val_fold)
            fold_rmse = root_mean_squared_error(y_val_fold, fold_pred)
            cv_loss += fold_rmse
            results["cv_loss"].append(fold_rmse)
        cv_loss /= self.cv.get_n_splits()
        results["avg_loss"] = cv_loss
        if refit:
            self.fitted_model = model.fit(X, y)
        return results

    def get_predicts(self, test_path, model_path=None, predict_with="refit"):
        """Predict test prices in dollars.

        predict_with: "refit" uses the model trained on all rows (or model_path),
        "folds" averages the CV fold models in log space, "first_fold" uses only
        the first fold model (trained on 80% of the rows).
        """
        if predict_with == "refit":
            if self.fitted_model is None and model_path is None:
                raise NotFittedError(
                    "No trained model: call fit_single_sklearn_model(refit=True) first or pass model_path"
                )
            models = [load(resolve_path(model_path)) if model_path else self.fitted_model]
        elif predict_with in ("folds", "first_fold"):
            if not self.fold_models:
                raise NotFittedError("No fold models: call fit_single_sklearn_model() first")
            models = self.fold_models if predict_with == "folds" else self.fold_models[:1]
        else:
            raise ValueError(f"Unknown predict_with: {predict_with}")

        ids, X_test = self.dataset.load_test(test_path)
        pred = np.mean([model.predict(X_test) for model in models], axis=0)
        pred = np.expm1(pred)
        submission = ids.with_columns(pl.Series(pred).alias("SalePrice"))
        return submission

    def save_model(self, path=None):
        """Dump the refit model with joblib (default: artefacts/models)."""
        if self.fitted_model is None:
            raise NotFittedError(
                "No trained model: call fit_single_sklearn_model() first"
            )

        path = resolve_path("artefacts/models") if path is None else resolve_path(path)
        path.mkdir(parents=True, exist_ok=True)
        dump(self.fitted_model, path)
        return path
