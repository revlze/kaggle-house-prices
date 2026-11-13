import polars as pl

categorical = [
    "MSSubClass",  # категориальный, потому что показывает типы из @data_description.txt прочитал
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]

numeric = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]

LOW_INFO = ("Street", "Utilities", "Condition2", "MiscFeature", "MiscVal")

# Порядковые шкалы из data_description.txt; "NA" = объект отсутствует -> 0.
# Null (настоящий пропуск, например KitchenQual в test) остаётся null для импутера.
ORDINAL_SCALES = {
    "quality": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "exposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "fin_type": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "garage_finish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "paved": {"N": 0, "P": 1, "Y": 2},
    "functional": {
        "Sal": 0,
        "Sev": 1,
        "Maj2": 2,
        "Maj1": 3,
        "Mod": 4,
        "Min2": 5,
        "Min1": 6,
        "Typ": 7,
    },
    "yes_no": {"N": 0, "Y": 1},
}

ORDINAL_QUALITY = (
    "ExterQual",
    "KitchenQual",
    "HeatingQC",
    "BsmtQual",
    "FireplaceQu",
    "GarageQual",
    "PoolQC",
)
# По цене немонотонны (TA дороже Gd/Ex) — для линейной модели проверяем отдельно.
ORDINAL_COND = ("ExterCond", "BsmtCond", "GarageCond")

ORDINAL_COLUMNS = {
    **{c: "quality" for c in ORDINAL_QUALITY + ORDINAL_COND},
    "BsmtExposure": "exposure",
    "BsmtFinType1": "fin_type",
    "BsmtFinType2": "fin_type",
    "GarageFinish": "garage_finish",
    "PavedDrive": "paved",
    "Functional": "functional",
    "CentralAir": "yes_no",
}

ORDINAL_ALL = tuple(ORDINAL_COLUMNS)
ORDINAL_NO_COND = tuple(c for c in ORDINAL_COLUMNS if c not in ORDINAL_COND)


AGGREGATES = {
    'TotalSF': pl.sum_horizontal('TotalBsmtSF', 'GrLivArea'),
    'TotalBath': pl.sum_horizontal(
        "FullBath", "BsmtFullBath",
        0.5 * pl.col("HalfBath"), 0.5 * pl.col("BsmtHalfBath"),
    ),
    'TotalPorchSF': pl.sum_horizontal(
        "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF",
    ),
    "HouseAge": (pl.col("YrSold") - pl.col("YearBuilt")).clip(lower_bound=0),
    "RemodAge": (pl.col("YrSold") - pl.col("YearRemodAdd")).clip(lower_bound=0)
}

variants = {
    "base": {"extra": [], "drop": [], "ordinal": ()},
    "pruned": {"extra": [], "drop": LOW_INFO, "ordinal": ()},
    "pruned_ordinal": {"extra": [], "drop": LOW_INFO, "ordinal": ORDINAL_ALL},
    "pruned_ordinal_no_cond": {
        "extra": [],
        "drop": LOW_INFO,
        "ordinal": ORDINAL_NO_COND,
    },
    "pruned_ordinal_agg": {"extra": tuple(AGGREGATES), "drop": LOW_INFO, "ordinal": ORDINAL_ALL},
    "pruned_ordinal_no_cond_agg": {
        "extra": tuple(AGGREGATES),
        "drop": LOW_INFO,
        "ordinal": ORDINAL_NO_COND,
    },
}


def make_features(df, feature_set, drop=()):
    variant = variants[feature_set]
    drop = set(variant["drop"]) | set(drop)
    unknown = drop - set(numeric) - set(categorical)
    if unknown:
        raise ValueError(f"Unknown features to drop: {sorted(unknown)}")

    extra = list(variant["extra"])
    unknown_extra = set(extra) - set(AGGREGATES)
    if unknown_extra:
        raise ValueError(f"Unknown features to agg in extra: {unknown_extra}")

    df = df.with_columns(AGGREGATES[name].alias(name) for name in extra)

    ordinal = [c for c in variant.get("ordinal", ()) if c not in drop]
    not_categorical = set(ordinal) - set(categorical)
    if not_categorical:
        raise ValueError(
            f"Ordinal columns must be categorical: {sorted(not_categorical)}"
        )

    # replace_strict падает на значении вне шкалы — опечатка в mapping не пройдёт молча.
    df = df.with_columns(
        pl.col(c).replace_strict(
            ORDINAL_SCALES[ORDINAL_COLUMNS[c]], return_dtype=pl.Int64
        )
        for c in ordinal
    )

    num = [c for c in numeric if c not in drop] + ordinal + extra
    cat = [c for c in categorical if c not in drop and c not in ordinal]
    return df.select(num + cat), num, cat
