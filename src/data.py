import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split

from pathlib import Path

from utils import resolve_path
from features import make_features


NA_ABSENCE_COLUMNS = (
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
)


class Dataset:
    def __init__(self, train_path, test_path, feature_set, seed, train_size=0.8):
        self.train_path = resolve_path(train_path)
        self.test_path = resolve_path(test_path)

        self.train_dataset = load_data(self.train_path)
        self.feature_set = feature_set
        self.seed = seed
        self.train_size = train_size
        self.X, self.numeric_columns, self.categorical_columns = make_features(
            self.train_dataset, self.feature_set
        )
        self.y = self.train_dataset["SalePrice"]

        self.X_train, self.X_holdout, self.y_trian, self.y_holdout = train_test_split(
            self.X,
            self.y,
            train_size=self.train_size,
            stratify=self.y,
            random_state=seed,
        )


def load_data(path):
    """Keep NA as an absence category where specified by the data dictionary."""
    columns = pl.read_csv(path, n_rows=0, infer_schema=False).columns
    null_values = {
        column: "NA" for column in columns if column not in NA_ABSENCE_COLUMNS
    }
    return pl.read_csv(path, null_values=null_values)
