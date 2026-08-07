"""
data_loader.py
==============
Streamlit-cached loaders for the dataset and model artifacts, so every page
of the dashboard reads from disk once per session instead of on every
rerun/widget interaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluation import load_model_comparison, load_training_history
from src.feature_engineering import engineer_features
from src.preprocessing import clean_dataset, encode_gender, load_raw_dataset
from src.prediction import available_models
from src.segmentation import DEFAULT_N_CLUSTERS, persona_summary, run_kmeans_segmentation


@st.cache_data(show_spinner=False)
def get_raw_dataset() -> pd.DataFrame:
    """Return the untouched dataset (for CSV download / row counts)."""
    return load_raw_dataset()


@st.cache_data(show_spinner=False)
def get_enriched_dataset() -> pd.DataFrame:
    """Return a cleaned + engineered (but NOT scaled) dataset for EDA /
    Customer Analytics charts. Gender is label-encoded to 0/1 but a human
    readable copy is preserved as 'Gender_Label' for charting.
    """
    df = load_raw_dataset()
    df = clean_dataset(df)
    df["Gender_Label"] = df["Gender"]  # readable copy before encoding
    encoded_df, _ = encode_gender(df.drop(columns=["Gender_Label"]))
    encoded_df["Gender_Label"] = df["Gender_Label"]
    encoded_df = engineer_features(encoded_df)
    return encoded_df


@st.cache_data(show_spinner=False)
def get_model_comparison() -> pd.DataFrame:
    return load_model_comparison()


@st.cache_data(show_spinner=False)
def get_training_history():
    return load_training_history()


@st.cache_data(show_spinner=False)

def get_available_models():
    """
    Return all supported models.
    Show unavailable models with a suffix.
    """

    models = []

    for name, path in config.MODEL_PATHS.items():
        if Path(path).exists():
            models.append(name)
        else:
            models.append(f"{name} (Not Available)")

    return models


def models_are_ready():
    """
    At least one model must exist.
    """

    return any(
        Path(path).exists()
        for path in config.MODEL_PATHS.values()
    )


@st.cache_data(show_spinner=False)
def get_segmented_dataset(n_clusters: int = DEFAULT_N_CLUSTERS):
    """Return (segmented_df, cluster_profile_df, persona_summary_df) built by
    running KMeans customer segmentation on the raw dataset. Cached so the
    clustering only runs once per session per ``n_clusters`` value.
    """
    df = get_raw_dataset()
    segmented_df, profile_df = run_kmeans_segmentation(df, n_clusters=n_clusters)
    persona_df = persona_summary(segmented_df)
    return segmented_df, profile_df, persona_df


def dataset_is_ready() -> bool:
    return config.DATASET_PATH.exists()


def models_are_ready() -> bool:
    return len(available_models()) > 0
