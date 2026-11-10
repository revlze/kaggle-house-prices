import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from catboost import CatBoostRegressor

    return (
        ColumnTransformer,
        KFold,
        LinearRegression,
        OneHotEncoder,
        Path,
        Pipeline,
        SimpleImputer,
    )


@app.cell
def _():
    from src.training import fit_single_sklearn_model
    from src.utils import set_seed
    from src.data import Dataset

    return Dataset, fit_single_sklearn_model, set_seed


@app.cell
def _():
    RUN_BASELINE= True
    RUN_TEST_DROP_CANDIDATES = True
    return RUN_BASELINE, RUN_TEST_DROP_CANDIDATES


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

    def make_model(dataset):
        return Pipeline([
            ('preprocessor', ColumnTransformer([
                ('numeric', SimpleImputer(strategy='median'), dataset.numeric_columns),
                ('categorical', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ])),
            ('regressor', LinearRegression()),
        ])

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return cv, dataset, make_model, seed, train_path


@app.cell
def _(RUN_BASELINE, cv, dataset, fit_single_sklearn_model, make_model, seed):
    if RUN_BASELINE:
        _results = fit_single_sklearn_model(make_model(dataset), cv, dataset, seed)
        print(_results['avg_loss'])
        print(_results['cv_loss'])

    return


@app.cell
def _(
    Dataset,
    RUN_TEST_DROP_CANDIDATES,
    cv,
    fit_single_sklearn_model,
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
            _results = fit_single_sklearn_model(make_model(dataset_dropped), cv, dataset_dropped, seed)
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


if __name__ == "__main__":
    app.run()
