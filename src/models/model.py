from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(dataset, scale=False):
    """Impute numeric columns (optionally standardize) and one-hot encode categoricals."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), dataset.numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                dataset.categorical_columns,
            ),
        ]
    )


def make_model(dataset, regressor, scale=False):
    """Wrap a regressor into a pipeline with preprocessing built for this dataset."""
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(dataset, scale)),
            ("regressor", regressor),
        ]
    )
