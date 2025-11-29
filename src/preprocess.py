import os
import logging
from .config import LOG_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Logging setup
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop duplicates
    - Map target to 0/1
    - Drop rows with missing target
    """
    df = df.copy()
    logger.info("Starting data cleaning.")

    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - df.shape[0]} duplicate rows.")

    # Map booking_status: Canceled -> 1, Not_Canceled -> 0
    if df[TARGET_COLUMN].dtype == "object":
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
            {"Canceled": 1, "Not_Canceled": 0}
        )

    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    logger.info("Data cleaning done.")
    return df

def treat_outliers_iqr(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Outlier treatment using IQR capping.
    Default columns: 'lead_time', 'avg_price_per_room'
    """
    df = df.copy()
    if cols is None:
        cols = ["lead_time", "avg_price_per_room"]

    logger.info(f"Starting outlier treatment for columns: {cols}")
    for col in cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not present, skipping.")
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.where(df[col] < lower, lower,
                           np.where(df[col] > upper, upper, df[col]))
        logger.info(f"Capped outliers in {col} to [{lower}, {upper}]")

    logger.info("Outlier treatment completed.")
    return df

def get_preprocessor(df: pd.DataFrame):
    """
    Create ColumnTransformer for numeric and categorical features.
    """
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET_COLUMN in numeric_features:
        numeric_features.remove(TARGET_COLUMN)

    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features

def split_data(df: pd.DataFrame):
    """
    Train-test split with stratification.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        f"Split data into train: {X_train.shape}, test: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test
