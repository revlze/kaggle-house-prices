import polars as pl
from sklearn.model_selection import train_test_split

from .features import make_features
from .utils import resolve_path

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
    """Train features and log1p target for one feature set, plus an 80/20 holdout split."""

    def __init__(self, train_path, feature_set, seed, train_size=0.8, drop=()):
        self.train_path = resolve_path(train_path)

        self.train_dataset = remove_outliers(load_data(self.train_path))
        self.feature_set = feature_set
        self.seed = seed
        self.train_size = train_size
        self.drop = tuple(drop)
        self.X, self.numeric_columns, self.categorical_columns = make_features(
            self.train_dataset, self.feature_set, self.drop
        )

        # Target is log1p(price), so RMSE on it is the competition metric;
        # predictions are mapped back with expm1 in Training.get_predicts.
        self.y = self.train_dataset["SalePrice"].log1p()

        self.X_train, self.X_holdout, self.y_train, self.y_holdout = train_test_split(
            self.X,
            self.y,
            train_size=self.train_size,
            random_state=seed,
        )

    def load_test(self, test_path):
        """Return test Ids and test features built with the same feature set and drops."""
        test_path = resolve_path(test_path)
        test_dataset = load_data(test_path)
        X_test, _, _ = make_features(test_dataset, self.feature_set, self.drop)
        return test_dataset.select("Id"), X_test


def load_data(path):
    """Keep NA as an absence category where specified by the data dictionary."""
    columns = pl.read_csv(path, n_rows=0, infer_schema=False).columns
    null_values = {
        column: "NA" for column in columns if column not in NA_ABSENCE_COLUMNS
    }
    return pl.read_csv(path, null_values=null_values)


def remove_outliers(df):
    """Drop the two train houses with huge living area but a low price.

    They are atypical sales (GrLivArea > 4000 sq ft, SalePrice < 300k) and pull
    linear models and the MLP far off; the two large houses priced over 700k
    are plausible and kept. Applied to train only: test needs a prediction per Id.
    """
    return df.filter(~((pl.col("GrLivArea") > 4000) & (pl.col("SalePrice") < 300_000)))


def save_submission(submission: pl.DataFrame, name: str):
    """Write the submission to submissions/submission_<name>.csv and return the path."""
    path = resolve_path("submissions")
    path.mkdir(exist_ok=True)
    path_csv = path / f"submission_{name}.csv"
    submission.write_csv(file=path_csv)
    return path_csv
