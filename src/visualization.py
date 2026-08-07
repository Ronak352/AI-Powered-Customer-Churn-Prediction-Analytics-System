"""
visualization.py
=================
Reusable Plotly chart builders shared across every Streamlit dashboard page.
Centralizing chart construction here keeps ``pages/*.py`` thin (just data
selection + layout) and guarantees a consistent visual style throughout the
app (colors, fonts, hover templates).

All functions return a ``plotly.graph_objects.Figure`` — the caller is
responsible for calling ``st.plotly_chart(fig, use_container_width=True)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent brand palette used across the whole dashboard.
PRIMARY_COLOR = "#4F8BF9"
CHURN_COLOR = "#EF553B"
NO_CHURN_COLOR = "#00CC96"
RISK_COLORS = {"Low": "#00CC96", "Medium": "#FFA15A", "High": "#EF553B"}
TEMPLATE = "plotly_white"


def _apply_common_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
    )
    return fig


def histogram(df: pd.DataFrame, column: str, title: str, color: Optional[str] = None, nbins: int = 30) -> go.Figure:
    """Distribution histogram with an optional KDE-like smoothed overlay via marginal box."""
    fig = px.histogram(
        df, x=column, nbins=nbins, color=color, marginal="box",
        color_discrete_sequence=[PRIMARY_COLOR] if color is None else None,
        template=TEMPLATE,
    )
    return _apply_common_layout(fig, title)


def count_bar(df: pd.DataFrame, column: str, title: str, color: Optional[str] = None) -> go.Figure:
    """Bar chart of value counts for a categorical column."""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", color=color or column, text="count", template=TEMPLATE)
    fig.update_traces(textposition="outside")
    return _apply_common_layout(fig, title)


def pie_chart(df: pd.DataFrame, column: str, title: str, hole: float = 0.0) -> go.Figure:
    """Pie (hole=0) or donut (hole>0) chart of category proportions."""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.pie(counts, names=column, values="count", hole=hole, template=TEMPLATE)
    return _apply_common_layout(fig, title)


def donut_chart(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    return pie_chart(df, column, title, hole=0.5)


def box_plot(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.box(df, x=x, y=y, color=x, template=TEMPLATE)
    return _apply_common_layout(fig, title)


def violin_plot(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.violin(df, x=x, y=y, color=x, box=True, points=False, template=TEMPLATE)
    return _apply_common_layout(fig, title)


def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, template=TEMPLATE, opacity=0.6)
    return _apply_common_layout(fig, title)


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    corr = df.select_dtypes(include="number").corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=TEMPLATE
    )
    return _apply_common_layout(fig, title)


def churn_rate_by_category(df: pd.DataFrame, category_col: str, target_col: str, title: str) -> go.Figure:
    """Grouped bar chart of churn rate (%) per category — used for
    age/geography/gender-wise churn business insights.
    """
    rate = df.groupby(category_col)[target_col].mean().reset_index()
    rate[target_col] = (rate[target_col] * 100).round(2)
    fig = px.bar(
        rate, x=category_col, y=target_col, text=target_col,
        color=target_col, color_continuous_scale="Reds", template=TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_yaxes(title="Churn Rate (%)")
    return _apply_common_layout(fig, title)


def gauge_chart(value: float, title: str = "Churn Probability", max_value: float = 100) -> go.Figure:
    """Speedometer-style gauge for a single prediction's risk percentage."""
    if value <= 1:
        value = value * 100
    color = RISK_COLORS["High"] if value >= 60 else RISK_COLORS["Medium"] if value >= 30 else RISK_COLORS["Low"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#e8f8f2"},
                    {"range": [30, 60], "color": "#fff2e0"},
                    {"range": [60, 100], "color": "#fde8e5"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": value},
            },
        )
    )
    fig.update_layout(template=TEMPLATE, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def kpi_delta_card_data(current: float, previous: Optional[float] = None):
    """Helper returning (value, delta) for use with st.metric — not a chart,
    but grouped here since it's part of the shared KPI vocabulary.
    """
    delta = None if previous is None else current - previous
    return current, delta


def radar_chart(categories: Sequence[str], values: Sequence[float], title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(values) + [values[0]], theta=list(categories) + [categories[0]], fill="toself"))
    fig.update_layout(template=TEMPLATE, title=title, polar=dict(radialaxis=dict(visible=True)))
    return fig


def treemap_chart(df: pd.DataFrame, path: List[str], values: Optional[str], title: str) -> go.Figure:
    fig = px.treemap(df, path=path, values=values, template=TEMPLATE)
    return _apply_common_layout(fig, title)


def sunburst_chart(df: pd.DataFrame, path: List[str], values: Optional[str], title: str) -> go.Figure:
    fig = px.sunburst(df, path=path, values=values, template=TEMPLATE)
    return _apply_common_layout(fig, title)


def roc_curve_figure(fpr, tpr, auc_value: float, title: str = "ROC Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc_value:.3f})", line=dict(color=PRIMARY_COLOR, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="gray")))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _apply_common_layout(fig, title)


def precision_recall_figure(precision, recall, title: str = "Precision-Recall Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", fill="tozeroy", line=dict(color=PRIMARY_COLOR, width=3)))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return _apply_common_layout(fig, title)


def confusion_matrix_figure(cm, labels: Sequence[str] = ("No Churn", "Churn"), title: str = "Confusion Matrix") -> go.Figure:
    fig = px.imshow(
        cm, text_auto=True, x=list(labels), y=list(labels),
        color_continuous_scale="Blues", template=TEMPLATE,
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    return _apply_common_layout(fig, title)


def training_history_figure(history: dict, metric: str = "accuracy", title: str = "") -> go.Figure:
    fig = go.Figure()
    if metric in history:
        fig.add_trace(go.Scatter(y=history[metric], mode="lines", name=f"Train {metric}"))
    val_key = f"val_{metric}"
    if val_key in history:
        fig.add_trace(go.Scatter(y=history[val_key], mode="lines", name=f"Validation {metric}"))
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title=metric.capitalize())
    return _apply_common_layout(fig, title or f"ANN {metric.capitalize()} over Epochs")


def pairplot_matrix(df: pd.DataFrame, dimensions: List[str], color: Optional[str] = None, title: str = "Pairplot") -> go.Figure:
    """Plotly scatter-matrix (SPLOM) — the interactive, dashboard-friendly
    equivalent of a seaborn pairplot, showing every pairwise relationship
    between the given numeric ``dimensions`` at once.
    """
    fig = px.scatter_matrix(
        df, dimensions=dimensions, color=color, template=TEMPLATE, opacity=0.6,
        color_discrete_sequence=[NO_CHURN_COLOR, CHURN_COLOR] if color else None,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4))
    fig.update_layout(height=150 * len(dimensions) + 100)
    return _apply_common_layout(fig, title)


def lift_gain_curves(y_true, y_proba, n_bins: int = 10):
    """Compute cumulative lift and gain curve data.

    Sorts customers by predicted churn probability (highest risk first),
    splits them into ``n_bins`` equal-sized deciles, and for each decile
    computes the cumulative % of actual churners captured (gain) and how
    many times better than random that is (lift).

    Returns:
        DataFrame with columns: Decile, % of Customers, % of Churners Captured
        (Gain), Lift.
    """
    order = np.argsort(-np.asarray(y_proba))
    y_sorted = np.asarray(y_true)[order]
    n = len(y_sorted)
    total_positives = y_sorted.sum()

    rows = []
    for i in range(1, n_bins + 1):
        cutoff = int(np.ceil(n * i / n_bins))
        captured = y_sorted[:cutoff].sum()
        pct_customers = cutoff / n * 100
        pct_captured = (captured / total_positives * 100) if total_positives > 0 else 0
        lift = (pct_captured / pct_customers) if pct_customers > 0 else 0
        rows.append(
            {
                "Decile": i,
                "% of Customers Contacted": round(pct_customers, 1),
                "% of Churners Captured": round(pct_captured, 1),
                "Lift": round(lift, 2),
            }
        )
    return pd.DataFrame(rows)


def gain_curve_figure(curve_df: pd.DataFrame, title: str = "Cumulative Gain Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df["% of Customers Contacted"], y=curve_df["% of Churners Captured"],
            mode="lines+markers", name="Model", line=dict(color=PRIMARY_COLOR, width=3),
        )
    )
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random Baseline", line=dict(dash="dash", color="gray")))
    fig.update_xaxes(title="% of Customers Contacted (ranked by risk)")
    fig.update_yaxes(title="% of Churners Captured")
    return _apply_common_layout(fig, title)


def lift_curve_figure(curve_df: pd.DataFrame, title: str = "Lift Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=curve_df["Decile"], y=curve_df["Lift"], marker_color=PRIMARY_COLOR, name="Lift"))
    fig.add_trace(go.Scatter(x=curve_df["Decile"], y=[1] * len(curve_df), mode="lines", name="Random Baseline (Lift=1)", line=dict(dash="dash", color="gray")))
    fig.update_xaxes(title="Decile (1 = highest predicted risk)")
    fig.update_yaxes(title="Lift")
    return _apply_common_layout(fig, title)


def force_plot_figure(feature_names: Sequence[str], shap_values: Sequence[float], base_value: float, prediction_value: float, title: str = "SHAP Force Plot") -> go.Figure:
    """A Plotly-native rendering of a SHAP force plot: a single horizontal
    axis from the base value to the final prediction, with each feature
    shown as a colored segment pushing the prediction up (red, increases
    churn risk) or down (blue/green, decreases churn risk) — the same
    visual idea as ``shap.plots.force`` but rendered with Plotly so it
    matches the rest of the dashboard's theme and works without JS shap
    bundles.
    """
    pairs = sorted(zip(feature_names, shap_values), key=lambda p: abs(p[1]), reverse=True)
    fig = go.Figure()
    running = base_value
    for name, val in pairs:
        color = CHURN_COLOR if val > 0 else NO_CHURN_COLOR
        fig.add_trace(
            go.Bar(
                x=[val], y=["Prediction"], orientation="h", base=running,
                marker_color=color, name=f"{name} ({val:+.3f})",
                hovertemplate=f"{name}<br>impact: {val:+.3f}<extra></extra>",
                showlegend=True,
            )
        )
        running += val

    fig.add_vline(x=base_value, line_dash="dot", line_color="gray", annotation_text=f"base value {base_value:.3f}")
    fig.add_vline(x=prediction_value, line_dash="solid", line_color="black", annotation_text=f"prediction {prediction_value:.3f}")
    fig.update_layout(barmode="stack", height=280, showlegend=True)
    fig.update_xaxes(title="Model output (churn probability contribution)")
    fig.update_yaxes(title="")
    return _apply_common_layout(fig, title)


def pipeline_diagram_figure() -> go.Figure:
    """Static workflow diagram illustrating the end-to-end AI/ML pipeline,
    from raw data to a business decision. Pure Plotly shapes/annotations —
    no external graph library required.
    """
    stages = [
        ("Raw Data", "Churn_Modelling.csv\n10,000 customers"),
        ("Preprocessing", "Clean · Encode\nGender & Geography"),
        ("Feature\nEngineering", "balance_to_salary\ntenure_by_age"),
        ("Train/Test\nSplit", "80/20 split\nstratified"),
        ("Model\nTraining", "LogReg · KNN · NB\nRF · XGBoost · ANN"),
        ("Evaluation", "Accuracy · F1\nROC AUC · Lift"),
        ("Explainability", "SHAP global &\nlocal explanations"),
        ("Deployment", "Streamlit + FastAPI\nBatch & single predict"),
        ("Business\nAction", "Risk scoring →\nRetention playbooks"),
    ]

    n = len(stages)
    box_w, gap = 1.0, 0.35
    fig = go.Figure()
    colors_cycle = [PRIMARY_COLOR, "#7C6FE0", "#00B4A2"]

    for i, (title, subtitle) in enumerate(stages):
        x0 = i * (box_w + gap)
        x1 = x0 + box_w
        color = colors_cycle[i % len(colors_cycle)]
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1,
            line=dict(color=color, width=2), fillcolor=color, opacity=0.15,
        )
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=f"<b>{title}</b>", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=(x0 + x1) / 2, y=0.32, text=subtitle.replace("\n", "<br>"), showarrow=False, font=dict(size=9, color="gray"))
        if i < n - 1:
            fig.add_annotation(
                x=x1 + gap / 2, y=0.5, ax=x1, ay=0.5, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowcolor="gray",
            )

    fig.update_xaxes(visible=False, range=[-0.2, n * (box_w + gap)])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(template=TEMPLATE, height=260, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return fig


def model_comparison_bar(comparison_df: pd.DataFrame, metric: str = "F1 Score", title: str = "") -> go.Figure:
    fig = px.bar(
        comparison_df.sort_values(metric, ascending=True), x=metric, y="Model", orientation="h",
        color=metric, color_continuous_scale="Blues", text=metric, template=TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return _apply_common_layout(fig, title or f"Model Comparison — {metric}")
