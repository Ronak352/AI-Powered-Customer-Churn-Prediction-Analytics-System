"""
ui.py
=====
Shared UI building blocks for the Streamlit dashboard: global CSS, KPI
cards, page headers, and navigation cards. Every page calls
``inject_global_css()`` once at the top so styling stays consistent without
duplicating CSS in every file.

The CSS uses Streamlit's own CSS variables (``var(--...)``) wherever
possible so the look adapts automatically to the user's light/dark theme
instead of hardcoding colors that would clash in dark mode.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

try:
    from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder, GridUpdateMode

    AGGRID_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AGGRID_AVAILABLE = False

GLOBAL_CSS = """
<style>
/* ---------- KPI Cards ---------- */
.kpi-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: left;
    transition: transform 0.15s ease;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-label { font-size: 0.85rem; opacity: 0.7; margin-bottom: 4px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; }
.kpi-sub { font-size: 0.78rem; opacity: 0.6; }

/* ---------- Nav Cards (Home page) ---------- */
.nav-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 16px;
    padding: 22px;
    height: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.nav-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.nav-card-icon { font-size: 2rem; margin-bottom: 8px; }
.nav-card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }
.nav-card-desc { font-size: 0.85rem; opacity: 0.75; }

/* ---------- Risk Badges ---------- */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
}
.risk-high   { background: #fde8e5; color: #c0392b; }
.risk-medium { background: #fff2e0; color: #b9770e; }
.risk-low    { background: #e8f8f2; color: #1e8449; }

/* ---------- Page Header ---------- */
.page-header {
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(128,128,128,0.15);
    margin-bottom: 18px;
}
.page-header h1 { margin-bottom: 0px; }
.page-header p { opacity: 0.75; margin-top: 2px; }
</style>
"""


def inject_global_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    """


def render_kpi_row(items: list[tuple[str, str, str]]) -> None:
    """Render a responsive row of KPI cards.

    Args:
        items: list of (label, value, sub) tuples.
    """
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        with col:
            st.markdown(kpi_card(label, value, sub), unsafe_allow_html=True)


def nav_card(icon: str, title: str, desc: str) -> str:
    return f"""
        <div class="nav-card">
            <div class="nav-card-icon">{icon}</div>
            <div class="nav-card-title">{title}</div>
            <div class="nav-card-desc">{desc}</div>
        </div>
    """


def risk_badge(risk_level: str) -> str:
    css_class = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(risk_level, "risk-medium")
    return f'<span class="risk-badge {css_class}">{risk_level} Risk</span>'


def empty_state(message: str, icon: str = "ℹ️") -> None:
    st.info(f"{icon} {message}")


def interactive_table(
    df: pd.DataFrame,
    key: Optional[str] = None,
    height: int = 420,
    selectable: bool = False,
    page_size: int = 15,
):
    """Render a dataframe as an interactive AgGrid table (sortable,
    filterable, resizable, column-pinnable, with client-side pagination) if
    ``streamlit-aggrid`` is installed; otherwise fall back to a plain
    ``st.dataframe`` so every page keeps working without the extra
    dependency.

    Args:
        df: Data to display.
        key: Unique Streamlit widget key (required if multiple grids are on
            the same page).
        height: Grid height in pixels.
        selectable: If True, enables single-row selection and returns the
            selected row (as a DataFrame) when using AgGrid.
        page_size: Rows per page for AgGrid's built-in pagination.

    Returns:
        The selected row(s) DataFrame if ``selectable=True`` and AgGrid is
        available and a row was selected, otherwise ``None``.
    """
    if not AGGRID_AVAILABLE:
        st.dataframe(df, use_container_width=True, height=height)
        st.caption("ℹ️ Install `streamlit-aggrid` for sortable/filterable interactive tables: `pip install streamlit-aggrid`")
        return None

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
    if selectable:
        gb.configure_selection(selection_mode="single", use_checkbox=False)
    grid_options = gb.build()

    result = AgGrid(
        df,
        gridOptions=grid_options,
        height=height,
        update_mode=GridUpdateMode.SELECTION_CHANGED if selectable else GridUpdateMode.NO_UPDATE,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        theme="streamlit",
        key=key,
        allow_unsafe_jscode=False,
    )

    if selectable:
        selected = result.get("selected_rows")
        if selected is not None and len(selected) > 0:
            return pd.DataFrame(selected)
    return None


def models_missing_warning() -> None:
    st.warning(
        "⚠️ No trained models found yet. Run `python -m src.training` from the "
        "project root to train and save every model before using this page.",
        icon="⚠️",
    )
