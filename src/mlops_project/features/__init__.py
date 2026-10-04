"""Train-only fitted preprocessing; serving loads the saved pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_project.data.policy import FEATURES, validate_rows

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", *FEATURES[5:11]]


def build_transformer():
    # No clipping/imputation: retain legitimate negative balances and rare observations.
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), [f for f in FEATURES if f not in CATEGORICAL]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator):
    """P2 fits on train only and saves this entire object; P3 only calls predict."""
    return Pipeline([("features", build_transformer()), ("model", estimator)])


def serving_frame(records):
    import pandas as pd

    if validate_rows(records, training=False):
        raise ValueError("Invalid credit-default serving records")
    return pd.DataFrame(records, columns=FEATURES)


def build_preprocessor(*, feature_columns):
    """Factory P2 calls through `preprocessor_factory`; returns an unfitted transformer."""
    if list(feature_columns) != list(FEATURES):
        raise ValueError("Preprocessor expects the reviewed feature order")
    return build_transformer()
