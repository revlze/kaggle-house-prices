import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, ElasticNet
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from catboost import CatBoostRegressor

    return (
        CatBoostRegressor,
        ColumnTransformer,
        KFold,
        LinearRegression,
        OneHotEncoder,
        Path,
        Pipeline,
        RidgeCV,
        SimpleImputer,
        mo,
        np,
        pl,
    )


@app.cell
def _():
    from src.training import Training
    from src.utils import set_seed
    from src.data import Dataset, save_submission

    return Dataset, Training, save_submission, set_seed


@app.cell
def _():
    RUN_BASELINE = False
    RUN_TEST_DROP_CANDIDATES = False
    RUN_ORDINAL = False
    RUN_SUBMISSION = False
    return RUN_BASELINE, RUN_ORDINAL, RUN_SUBMISSION, RUN_TEST_DROP_CANDIDATES


@app.cell
def _(
    ColumnTransformer,
    Dataset,
    KFold,
    LinearRegression,
    OneHotEncoder,
    Path,
    Pipeline,
    SimpleImputer,
    set_seed,
):
    seed = 0xFACED
    base_path = Path('/Users/platon/Documents/kaggle-house')

    set_seed(seed)
    train_path = base_path / 'data/train.csv'

    dataset = Dataset(train_path=train_path, feature_set='base', seed=seed, train_size=0.8)

    def make_model(dataset, regressor=None):
        return Pipeline([
            ('preprocessor', ColumnTransformer([
                ('numeric', SimpleImputer(strategy='median'), dataset.numeric_columns),
                ('categorical', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ])),
            ('regressor', regressor if regressor is not None else LinearRegression()),
        ])

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return cv, dataset, make_model, seed, train_path


@app.cell
def _(RUN_BASELINE, Training, cv, dataset, make_model, seed):
    if RUN_BASELINE:
        _results = Training(make_model(dataset), cv, dataset, seed).fit_single_sklearn_model()
        print(_results['avg_loss'])
        print(_results['cv_loss'])
    return


@app.cell
def _(
    Dataset,
    RUN_TEST_DROP_CANDIDATES,
    Training,
    cv,
    make_model,
    seed,
    train_path,
):
    if RUN_TEST_DROP_CANDIDATES:
        from itertools import combinations

        candidates = ["Street", "Utilities", "Condition2", "MiscFeature", "MiscVal"]

        drop_sets = [
            subset
            for size in range(len(candidates) + 1)
            for subset in combinations(candidates, size)
        ]
        history = {}
        for drop_set in drop_sets:
            dataset_dropped = Dataset(train_path=train_path, feature_set='base', seed=seed, train_size=0.8, drop=drop_set)
            _results = Training(make_model(dataset_dropped), cv, dataset_dropped, seed).fit_single_sklearn_model()
            history[drop_set] = _results

        print(history)
    return


@app.cell
def _(mo):
    mo.md(r"""
    просто удалю всех `candidates = ["Street", "Utilities", "Condition2", "MiscFeature", "MiscVal"]` ради простоты

    потому что они не дают почти никакого сдвига на cv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Порядковые признаки в numeric

    Шкалы качества (`*Qual`, `HeatingQC`, `FireplaceQu`, `PoolQC`: Po..Ex = 1..5), `BsmtExposure`,
    `BsmtFinType1/2`, `GarageFinish`, `PavedDrive`, `Functional`, `CentralAir` переводятся в числа, NA = 0.

    - `pruned` — точка отсчёта, порядковые остаются категориями (one-hot)
    - `pruned_ordinal` — все 17 порядковых в numeric
    - `pruned_ordinal_no_cond` — без `ExterCond`, `BsmtCond`, `GarageCond`: по цене они немонотонны
      (TA дороже Gd/Ex), для линейной модели это может мешать

    Сравнение пофолдовое на одних и тех же фолдах: `diff` = лосс варианта − лосс `pruned` в том же фолде,
    `better` = в скольких фолдах из 5 вариант лучше.
    """)
    return


@app.cell
def _(
    CatBoostRegressor,
    Dataset,
    LinearRegression,
    RUN_ORDINAL,
    RidgeCV,
    Training,
    cv,
    make_model,
    np,
    pl,
    seed,
    train_path,
):
    if RUN_ORDINAL:
        _models = {
            'linear': lambda: LinearRegression(),
            'ridge': lambda: RidgeCV(alphas=np.logspace(-2, 3, 30)),
            'catboost': lambda: CatBoostRegressor(silent=True, random_seed=seed),
        }
        _variants = ['pruned', 'pruned_ordinal', 'pruned_ordinal_no_cond']

        ordinal_results = {}
        for _model_name, _make_regressor in _models.items():
            for _variant in _variants:
                _ds = Dataset(train_path=train_path, feature_set=_variant, seed=seed, train_size=0.8)
                ordinal_results[(_model_name, _variant)] = Training(
                    make_model(_ds, _make_regressor()), cv, _ds, seed
                ).fit_single_sklearn_model()

        _rows = []
        for (_model_name, _variant), _res in ordinal_results.items():
            _folds = np.array(_res['cv_loss'])
            _ref = np.array(ordinal_results[(_model_name, 'pruned')]['cv_loss'])
            _diff = _folds - _ref
            _rows.append({
                'model': _model_name,
                'variant': _variant,
                'avg': _folds.mean(),
                'std': _folds.std(),
                'diff_mean': _diff.mean(),
                'diff_std': _diff.std(),
                'better': int((_diff < 0).sum()),
            })
        ordinal_table = pl.DataFrame(_rows)
    else:
        ordinal_table = None
    ordinal_table
    return (ordinal_table,)


@app.cell
def _(ordinal_table):
    if ordinal_table is not None:
        ordinal_table.write_clipboard()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Сабмит
    """)
    return


@app.cell
def _(
    CatBoostRegressor,
    Dataset,
    RUN_SUBMISSION,
    Training,
    cv,
    make_model,
    save_submission,
    seed,
    train_path,
):
    if RUN_SUBMISSION:
        _feature_set = 'pruned_ordinal'
        _ds = Dataset(train_path=train_path, feature_set=_feature_set, seed=seed, train_size=0.8)
        _training = Training(make_model(_ds, CatBoostRegressor(silent=True, random_seed=seed)), cv, _ds, seed)
        _results = _training.fit_single_sklearn_model()
        print(_results['avg_loss'])
        save_submission(_training.get_predicts('data/test.csv'), f'catboost_{_feature_set}')
    return


if __name__ == "__main__":
    app.run()
