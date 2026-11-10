from sklearn.model_selection import KFold
from .utils import set_seed
from sklearn.base import clone
from .data import Dataset
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error

def fit_single_sklearn_model(model, cv, dataset, seed):
    set_seed(seed)
    model = clone(model)

    results = {
        'cv_loss': [],
        'avg_loss': None
    }
    X, y = dataset.X, dataset.y
    split_loop = tqdm(cv.split(X), total=cv.get_n_splits(), desc='CV')
    cv_loss = 0
    for i, (train_idx, val_idx) in enumerate(split_loop, start=1):
        fold_model = clone(model)
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
        fold_model.fit(X_train_fold, y_train_fold)
        fold_pred = fold_model.predict(X_val_fold)
        fold_rmse = root_mean_squared_error(y_val_fold, fold_pred)
        cv_loss += fold_rmse
        results['cv_loss'].append(fold_rmse)
    cv_loss /= cv.get_n_splits()
    results['avg_loss'] = cv_loss
    model.fit(X, y)
    return results
