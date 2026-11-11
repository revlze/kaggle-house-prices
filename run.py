import logging

import hydra
from hydra.utils import instantiate

from src.data import Dataset, save_submission
from src.models.model import make_model
from src.training import Training
from src.utils import set_seed

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    """Run CV for the configured model and feature set; optionally write a submission.

    Returns the mean CV loss so the Optuna sweeper can use it as the objective.
    """
    set_seed(cfg.seed)
    dataset = Dataset(cfg.train_path, cfg.feature_set, cfg.seed)
    model = make_model(dataset, instantiate(cfg.model.regressor), scale=cfg.model.scale)
    training = Training(model, instantiate(cfg.cv), dataset, cfg.seed)

    results = training.fit_single_sklearn_model(refit=cfg.submit)
    log.info(f"{cfg.model.name} | {cfg.feature_set} | CV {results['avg_loss']:.5f}")
    log.info(f"folds: {[round(loss, 5) for loss in results['cv_loss']]}")

    if cfg.submit:
        name = f"{cfg.model.name}_{cfg.feature_set}"
        path_csv = save_submission(training.get_predicts(cfg.test_path), name)
        log.info(f"submission saved: {path_csv}")

    return results['avg_loss']


if __name__ == "__main__":
    main()
