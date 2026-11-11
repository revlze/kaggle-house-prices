import numpy as np
from sklearn.model_selection import StratifiedKFold


class StratifiedRegressionKFold:
    """K-fold for a continuous target: stratify by quantile bins of y.

    sklearn's StratifiedKFold only accepts class labels, so y is cut into
    n_bins quantile bins (equal number of rows per bin) and the bins are used
    as labels. n_bins defaults to Sturges' rule: 1 + log2(n).
    """

    def __init__(self, n_splits=5, n_bins=None, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        y = np.asarray(y)
        n_bins = self.n_bins or int(1 + np.log2(len(y)))
        # inner quantile edges only: outer ones would put the max value in its own bin
        edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.digitize(y, edges)
        splitter = StratifiedKFold(self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return splitter.split(X, bins)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
