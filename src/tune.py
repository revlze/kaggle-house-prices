import optuna

from .training import Training


def tune(build_model, dataset, cv, seed, n_trials=20):
    def objective(trial):
        model = build_model(trial, dataset)
        results = Training(model, cv, dataset, seed).fit_single_sklearn_model()
        return results['avg_loss']

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    return study