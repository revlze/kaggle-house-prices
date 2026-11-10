categorical = [
    "MSSubClass", # категориальный, потому что показывает типы из @data_description.txt прочитал
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

variants = {
    "base": {"extra": [], "drop": []},
    "pruned": {"extra": [], "drop": LOW_INFO},
}


def make_features(df, feature_set, drop=()):
    variant = variants[feature_set]
    drop = set(variant["drop"]) | set(drop)
    unknown = drop - set(numeric) - set(categorical)
    if unknown:
        raise ValueError(f"Unknown features to drop: {sorted(unknown)}")

    extra = variant["extra"]
    # TODO: extend numeric and categorical with extra FE

    num = [c for c in numeric if c not in drop]
    cat = [c for c in categorical if c not in drop]
    return df.select(num + cat), num, cat
