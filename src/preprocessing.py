"""
preprocessing.py
=================
Data cleaning, encoding, and scaling utilities.

IMPORTANT: This module intentionally reproduces the exact preprocessing
logic from the original research notebook
(``notebooks/Bank_customer_churn_prediction.ipynb``) so that predictions
made by the deployed models stay perfectly consistent with how the models
were trained. Nothing from the original logic has been removed — it has
only been wrapped into reusable, testable, documented functions.

Original notebook steps reproduced here:
    1. Drop identifier columns: RowNumber, CustomerId, Surname
    2. Check for missing values and impute them only if present
       (median for numeric columns, mode for categorical columns)
    3. Label-encode Gender (Female=0, Male=1)
    4. One-hot encode Geography with drop_first=True
    5. Standard-scale all numeric features
"""

from __future__ import annotations

import json
import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_raw_dataset(path=None) -> pd.DataFrame:
    """Load the raw Churn_Modelling.csv dataset from disk.

    Args:
        path: Optional override path to a CSV file. Defaults to
            ``config.DATASET_PATH``.

    Returns:
        The raw dataframe, unmodified.
    """
    csv_path = path or config.DATASET_PATH
    logger.info("Loading raw dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop RowNumber, CustomerId, Surname — non-predictive identifiers."""
    cols_to_drop = [c for c in config.RAW_ID_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a per-column count of missing values (as in the notebook EDA)."""
    return df.isnull().sum()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values if any are present, otherwise leave the data untouched.

    The training dataset (Churn_Modelling.csv) does not contain nulls, but
    real-world data pushed through this pipeline later — a batch CSV upload,
    a new export, a manually edited file — might. Rather than assuming the
    data is always clean, this step checks first and only imputes columns
    that actually need it:

        - Numeric columns  -> filled with the column median (robust to
          outliers such as very high balances or salaries).
        - Categorical/text columns -> filled with the column mode (most
          frequent value).

    If no missing values are found, the dataframe is returned unchanged and
    no imputation logic runs at all.
    """
    df = df.copy()
    missing_counts = check_missing_values(df)
    columns_with_missing = missing_counts[missing_counts > 0]

    if columns_with_missing.empty:
        logger.info("No missing values detected — skipping imputation.")
        return df

    logger.info("Missing values detected in %d column(s), imputing...", len(columns_with_missing))
    for column in columns_with_missing.index:
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].median()
        else:
            mode_values = df[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)
        logger.info("  - '%s': filled %d missing value(s) with %s", column, columns_with_missing[column], fill_value)

    return df


def encode_gender(df: pd.DataFrame, encoder: LabelEncoder = None) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode the Gender column.

    Args:
        df: Dataframe containing a 'Gender' column with values in
            {'Male', 'Female'}.
        encoder: Optional pre-fit LabelEncoder (used at inference time).
            If None, a new encoder is fit on ``df`` (training time).

    Returns:
        (encoded dataframe, fitted encoder)
    """
    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
        df["Gender"] = encoder.fit_transform(df["Gender"])
    else:
        df["Gender"] = encoder.transform(df["Gender"])
    return df, encoder


def encode_geography(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """One-hot encode Geography with drop_first=True, matching the notebook.

    At inference time, this also guarantees that all expected dummy columns
    exist (filled with 0) even if a batch/single record does not contain
    every category, and saves/loads the category list to keep column order
    stable between training and inference.
    """
    df = df.copy()

    if training:
        df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
        geography_dummy_cols = [c for c in df.columns if c.startswith("Geography_")]
        with open(config.GEOGRAPHY_CATEGORIES_PATH, "w") as f:
            json.dump(geography_dummy_cols, f, indent=2)
        return df

    # Inference time: recreate the same dummy columns used in training.
    if not config.GEOGRAPHY_CATEGORIES_PATH.exists():
        raise FileNotFoundError(
            "geography_categories.json not found — run training.py first."
        )
    with open(config.GEOGRAPHY_CATEGORIES_PATH) as f:
        expected_cols = json.load(f)

    dummies = pd.get_dummies(df["Geography"], prefix="Geography")
    df = df.drop(columns=["Geography"])
    for col in expected_cols:
        df[col] = dummies[col] if col in dummies.columns else 0
    df = pd.concat([df, ], axis=1)  # no-op, keeps intent explicit
    return df


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training features."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_preprocessing_artifacts(gender_encoder: LabelEncoder, scaler: StandardScaler, feature_order: list) -> None:
    """Persist the gender encoder, scaler, and feature column order to disk."""
    joblib.dump(gender_encoder, config.GENDER_ENCODER_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    with open(config.FEATURE_ORDER_PATH, "w") as f:
        json.dump(feature_order, f, indent=2)
    logger.info("Saved gender_encoder.pkl, scaler.pkl, feature_order.json to %s", config.MODELS_DIR)


def load_preprocessing_artifacts():
    """Load the gender encoder, scaler, and feature order from disk.

    Returns:
        (gender_encoder, scaler, feature_order: list[str])
    """
    gender_encoder = joblib.load(config.GENDER_ENCODER_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    with open(config.FEATURE_ORDER_PATH) as f:
        feature_order = json.load(f)
    return gender_encoder, scaler, feature_order


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning step: drop identifier columns, then check for and
    handle missing values (only imputes if any are actually found).
    """
    df = drop_identifier_columns(df)
    df = handle_missing_values(df)
    return df
