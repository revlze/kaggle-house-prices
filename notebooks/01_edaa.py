import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    import polars as pl
    from skimpy import skim
    import sweetviz as sv
    from data_profiling import ProfileReport
    import pygwalker as pyg

    from pathlib import Path

    from src.data import load_data

    return Path, ProfileReport, load_data, mo, pl, plt, skim, sv


@app.cell
def _(Path):

    base_path = Path("/Users/platon/Documents/kaggle-house")

    curr_path = Path(__file__).parent
    html_path = curr_path/'htmls'
    html_path.mkdir(exist_ok=True)
    return base_path, html_path


@app.cell
def _(base_path, load_data):
    train_dataset = load_data(base_path / "data/train.csv")
    test_dataset = load_data(base_path / "data/test.csv")
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    train_dataset['SalePrice']
    return


@app.cell
def _(plt, train_dataset):
    train_dataset.to_pandas().hist(figsize=(16, 12), bins=30)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(skim, train_dataset):
    skimpy_summary = skim(train_dataset)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    skimpy — малочитабельно для столь обширного кол-ва фичей. но можно быстро пробежаться и посмотреьт еще и на категориальные
    """)
    return


@app.cell
def _(test_dataset, train_dataset):
    train_pd = train_dataset.to_pandas()
    test_pd = test_dataset.to_pandas()
    return test_pd, train_pd


@app.cell
def _(ProfileReport, html_path, train_pd):
    report_dp = ProfileReport(train_pd, title="House Prices EDA")
    report_dp.to_file(html_path/ "eda_profile.html")
    return


@app.cell
def _(sv, train_pd):
    report_sv = sv.analyze(train_pd, target_feat="SalePrice")
    return (report_sv,)


@app.cell
def _(html_path, report_sv):
    report_sv.show_html(html_path/ 'eda_profile_sv.html')
    return


@app.cell
def _(html_path, sv, test_pd, train_pd):
    report_compare = sv.compare(
        [train_pd.drop(columns=["SalePrice", "Id"]), "Train"],
        [test_pd.drop(columns=["Id"]), "Test"],
    )
    report_compare.show_html(html_path/ "eda_train_test.html")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - фича Alley — там чет 94% NA
    - фича Street — там чет 99% одного значения
    - фича Utilities — там чет 99% одного значения
    - фича LandSlop - 95% одного значения и еще два других значения по 4% и 1%
    - фича Condition2 — 99% одного значения и еще 7 других значений по 1%
    - фича RoofMatl — 98% одного значения и еще 7 других значений по 1%, но есть какая-то корреляция с SalePrice = 0.18
    - фича BsmtFinSF2 — чет просто странная, там 88% нулей, а остальные значения почти все разные, диапазон от 0 до 1474
    - фича BsmtHalfBath — там 94% нулей, а остальные странные тоже, думаю точно удалять стоит
    - фича FireplaceQu — там 47% NA, нужно чем-то заполнять, но как? может просто 0, а потом сделать категориальную фичу есть камин или нет
    - фича EnclosedPorch, 3SsnPorch, ScreenPorch — там >85% одно значение, а остальные раскиданы сильно, надо подумать, мб просто добавить фичу, есть веранда или нет, а потом сделать категориальную фичу, что за веранда
    - фича PoolArea — там 99% нулей, просто в категориальную фичу, есть бассейн или нет
    - фича PoolQC - там 99% NA нах
    - фича Fence — там 81% NA, нужно чем-то заполнять, но как? может просто "IDK", а потом сделать категориальную фичу есть забор или нет
    - фича MiscFeature - там 96% NA, как будто просто удалить
    - фича MiscVal — там 96.4% нулей, как будто делит
    - фича YrSold — чуть бесполезная фича вроде, там везде одна цена по сути 181к +- 4к, можно делит?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    чат жпт:


    | Признак | Моя оценка | Что делать |
    |---|---|---|
    | `Alley` | 94% NaN означают отсутствие доступа к переулку, а не неизвестное значение | Оставить, заполнить `NoAlley` |
    | `Street` | Всего 6 домов с гравийной дорогой | Кандидат на удаление |
    | `Utilities` | Только 1 дом отличается от остальных | Проверить удаление через CV |
    | `LandSlope` | 95% `Gtl`, но остальные 78 домов имеют иной уклон. Связь с `LotArea` — 0.55 | Оставить |
    | `Condition2` | Только 15 домов имеют значение, отличное от `Norm` | Кандидат на удаление |
    | `RoofMatl` | 98% `CompShg`, связь с ценой 0.18 | Оставить для проверки, возможно объединить редкие категории |
    | `BsmtFinSF2` | 89% нулей, остальные значения — площадь второй отделанной части подвала | Оставить числовым |
    | `BsmtHalfBath` | 94% нулей, количество дополнительных полуванных в подвале | Оставить, проверить объединение с другими ванными |
    | `FireplaceQu` | 47% NaN соответствуют домам без камина | Оставить, заполнить `NoFireplace` |
    | `EnclosedPorch`, `3SsnPorch`, `ScreenPorch` | Разные типы веранд, с большой долей нулей | Оставить, добавить агрегированные признаки |
    | `PoolArea` | Бассейн есть только у 7 домов | Проверить бинарный `HasPool` вместо площади |
    | `PoolQC` | Пропуски означают отсутствие бассейна | Заполнить `NoPool`, проверить полезность |
    | `Fence` | 81% NaN означают отсутствие забора | Оставить, заполнить `NoFence` |
    | `MiscFeature` | 96% NaN означают отсутствие дополнительных объектов | Кандидат на удаление |
    | `MiscVal` | 96.4% нулей, слабая линейная связь с ценой −0.02 | Кандидат на удаление, проверить вместе с `MiscFeature` |
    | `YrSold` | Средние цены по годам близки, но это не исключает временного эффекта | Пока оставить |
    """)
    return


@app.cell
def _(pl, train_dataset):
    train_dataset.filter(pl.col("GrLivArea") >= 3000).select("GrLivArea", "SalePrice").sort('GrLivArea')
    return


@app.cell
def _(pl, test_dataset):
    test_dataset.filter(pl.col("GrLivArea") > 4000)
    return


if __name__ == "__main__":
    app.run()
