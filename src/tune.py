import copy

import optuna
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .models.model import make_model
from .training import Training


def suggest(trial, name, spec):
    """Sample one hyperparameter from a search-space entry of the tune config."""
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )
    if kind == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown search space type '{kind}' for '{name}'")


def apply_params(cfg, params):
    """Return a copy of cfg with params written under model.regressor (dotted paths).

    The whole config is copied so interpolations like ${seed} keep resolving.
    """
    cfg = copy.deepcopy(cfg)
    for name, value in params.items():
        OmegaConf.update(cfg, f"model.regressor.{name}", value, force_add=True)
    return cfg


def tune(cfg, dataset):
    """Search cfg.tune.params with Optuna TPE, minimizing mean CV loss."""

    def objective(trial):
        params = {
            name: suggest(trial, name, spec) for name, spec in cfg.tune.params.items()
        }
        trial_cfg = apply_params(cfg, params)
        model = make_model(
            dataset, instantiate(trial_cfg.model.regressor), scale=cfg.model.scale
        )
        training = Training(model, instantiate(cfg.cv), dataset, cfg.seed)
        return training.fit_single_sklearn_model(refit=False)["avg_loss"]

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.tune.n_trials)
    return study
