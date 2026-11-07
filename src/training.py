from sklearn.model_selection import StratifiedKFold
from utils import set_seed
from sklearn.base import clone

def _fit_single_sklearn_model(model, cv, dataset, seed, epoches):
    model = clone(model)
    X_train, y_train = dataset.X_train, dataset.y_train
    X_holdout, y_holdout = dataset.X_val, dataset.y_val

    
