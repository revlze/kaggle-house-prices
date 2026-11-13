from sklearn.ensemble import StackingRegressor, VotingRegressor


def voting(estimators, **kwargs):
    """Build VotingRegressor from a {name: estimator} dict (readable in Hydra configs)."""
    return VotingRegressor(list(estimators.items()), **kwargs)


def stacking(estimators, **kwargs):
    """Build StackingRegressor from a {name: estimator} dict."""
    return StackingRegressor(list(estimators.items()), **kwargs)
