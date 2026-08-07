"""Data Integration Center: upload data, validate schema, and audit data quality."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from utils import ui
from utils.data_loader import dataset_is_ready, get_raw_dataset

st.set_page_config(page_title="Data Integration Center", page_icon="🧮", layout="wide")
ui.inject_global_css()
ui.page_header("Data Integration Center", "Upload new data, validate its schema, and audit data quality before it hits the pipeline.", "🧮")

tab_upload, tab_validation, tab_quality = st.tabs(["⬆️ Upload Data", "✅ Validation", "🧪 Data Quality"])

REQUIRED_COLUMNS = config.RAW_INPUT_COLUMNS + [config.TARGET_COLUMN]


def _quality_report(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    outlier_counts = {}
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((numeric_df[col] < lo) | (numeric_df[col] > hi)).sum())

    return {
        "shape": df.shape,
        "missing": df.isnull().sum(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str),
        "outliers": outlier_counts,
    }


# ==========================================================================
# UPLOAD DATA
# ==========================================================================
with tab_upload:
    st.subheader("Upload a Dataset")
    st.caption(
        "Upload a CSV with the same schema as the training dataset (raw churn "
        "feature columns, optionally including `Exited`). This does not "
        "overwrite the production dataset — it's a sandbox for validating "
        "new data before it's used anywhere else in the app."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="dic_uploader")

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.session_state["dic_uploaded_df"] = new_df
            st.success(f"Loaded `{uploaded.name}` — {new_df.shape[0]:,} rows × {new_df.shape[1]} columns.")
            ui.interactive_table(new_df.head(20), key="dic_preview_table")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read this file: {exc}")

    st.divider()
    if dataset_is_ready():
        st.caption("No upload yet? You can also validate the current production dataset below.")
        if st.button("Use current production dataset instead"):
            st.session_state["dic_uploaded_df"] = get_raw_dataset()
            st.rerun()

# ==========================================================================
# VALIDATION
# ==========================================================================
with tab_validation:
    active_df = st.session_state.get("dic_uploaded_df")

    if active_df is None:
        st.info("Upload a CSV (or load the production dataset) on the **Upload Data** tab first.")
    else:
        st.subheader("Schema Validation")

        present_cols = set(active_df.columns)
        required_cols = set(REQUIRED_COLUMNS)
        missing_cols = sorted(required_cols - present_cols)
        extra_cols = sorted(present_cols - required_cols - set(config.RAW_ID_COLUMNS))

        c1, c2, c3 = st.columns(3)
        c1.metric("Required Columns Present", f"{len(required_cols) - len(missing_cols)}/{len(required_cols)}")
        c2.metric("Missing Required Columns", len(missing_cols))
        c3.metric("Unexpected Extra Columns", len(extra_cols))

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            st.success("✅ All required columns are present.")

        if extra_cols:
            st.warning(f"⚠️ Extra columns not used by the model: {', '.join(extra_cols)}")

        st.markdown("##### Column-by-Column Check")
        check_rows = []
        for col in REQUIRED_COLUMNS:
            check_rows.append(
                {
                    "Column": col,
                    "Present": "✅" if col in present_cols else "❌",
                    "Dtype": str(active_df[col].dtype) if col in present_cols else "—",
                    "Nulls": int(active_df[col].isnull().sum()) if col in present_cols else "—",
                }
            )
        ui.interactive_table(pd.DataFrame(check_rows), key="dic_validation_table")

# ==========================================================================
# DATA QUALITY
# ==========================================================================
with tab_quality:
    active_df = st.session_state.get("dic_uploaded_df")

    if active_df is None:
        st.info("Upload a CSV (or load the production dataset) on the **Upload Data** tab first.")
    else:
        report = _quality_report(active_df)

        ui.render_kpi_row(
            [
                ("Rows", f"{report['shape'][0]:,}", ""),
                ("Columns", str(report["shape"][1]), ""),
                ("Duplicate Rows", f"{report['duplicates']:,}", f"{report['duplicates']/max(report['shape'][0],1)*100:.2f}%"),
                ("Total Missing Cells", f"{int(report['missing'].sum()):,}", ""),
            ]
        )

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Missing Values by Column")
            missing_df = report["missing"].rename("Missing Count").reset_index().rename(columns={"index": "Column"})
            missing_df["Missing %"] = (missing_df["Missing Count"] / len(active_df) * 100).round(2)
            ui.interactive_table(missing_df, key="dic_missing_table", height=300)
        with c2:
            st.markdown("##### Dtype Summary")
            dtype_df = report["dtypes"].rename("Dtype").reset_index().rename(columns={"index": "Column"})
            ui.interactive_table(dtype_df, key="dic_dtype_table", height=300)

        st.markdown("##### Outlier Summary (IQR method, numeric columns)")
        outlier_df = pd.DataFrame(
            [{"Column": k, "Outlier Count": v, "Outlier %": round(v / len(active_df) * 100, 2)} for k, v in report["outliers"].items()]
        ).sort_values("Outlier Count", ascending=False)
        ui.interactive_table(outlier_df, key="dic_outlier_table", height=300)

        if report["duplicates"] > 0:
            with st.expander(f"Show {report['duplicates']} duplicate rows"):
                ui.interactive_table(active_df[active_df.duplicated(keep=False)], key="dic_duplicate_rows")
