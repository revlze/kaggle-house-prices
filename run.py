import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data import Dataset, save_submission
from src.models.model import make_model
from src.training import Training
from src.tune import apply_params, tune
from src.utils import set_seed

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    """Run CV for the configured model and feature set; optionally write a submission.

    With +tune=<model>, Optuna first searches the tune config's space and the best
    parameters replace the model config before the final CV/refit/submission.
    Returns the mean CV loss.
    """
    set_seed(cfg.seed)
    dataset = Dataset(cfg.train_path, cfg.feature_set, cfg.seed)

    if "tune" in cfg:
        study = tune(cfg, dataset)
        log.info(f"tune: best CV {study.best_value:.5f} after {len(study.trials)} trials")
        log.info(f"tune: best params {study.best_params}")
        out_dir = Path(HydraConfig.get().runtime.output_dir)
        OmegaConf.save(OmegaConf.create(study.best_params), out_dir / "best_params.yaml")
        cfg = apply_params(cfg, study.best_params)

    model = make_model(dataset, instantiate(cfg.model.regressor), scale=cfg.model.scale)
    training = Training(model, instantiate(cfg.cv), dataset, cfg.seed)

    refit = cfg.submit and cfg.predict_with == "refit"
    results = training.fit_single_sklearn_model(refit=refit)
    log.info(f"{cfg.model.name} | {cfg.feature_set} | CV {results['avg_loss']:.5f}")
    log.info(f"folds: {[round(loss, 5) for loss in results['cv_loss']]}")

    if cfg.submit:
        name = f"{cfg.model.name}_{cfg.feature_set}_{cfg.name_exp}"
        if cfg.predict_with != "refit":
            name += f"_{cfg.predict_with}"
        submission = training.get_predicts(cfg.test_path, predict_with=cfg.predict_with)
        path_csv = save_submission(submission, name)
        log.info(f"submission saved: {path_csv}")

    return results['avg_loss']


if __name__ == "__main__":
    main()
