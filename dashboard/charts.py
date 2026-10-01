"""Reusable Plotly visualizations for the European Banking Churn Dashboard.

Follows modern design principles, consistent color palettes, responsive layouts,
and clear annotations.
"""
from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Theme Colors
COLOR_RETAINED = "#10B981"  # Emerald
COLOR_CHURNED = "#EF4444"   # Crimson Red
COLOR_PRIMARY = "#1E40AF"   # Deep Blue
COLOR_ACCENT = "#3B82F6"    # Sky Blue
COLOR_NEUTRAL = "#64748B"   # Slate Gray
COLOR_BG_CARD = "#F8FAFC"
PALETTE_CATEGORICAL = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]


def create_empty_figure(message: str = "No data available for current selection") -> go.Figure:
    """Return a clean empty figure with a centered informative message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color="#6B7280"),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    return fig


def plot_churn_donut(df: pd.DataFrame) -> go.Figure:
    """Plot Churn vs Retention Donut Chart."""
    if len(df) == 0:
        return create_empty_figure()

    churn_counts = df["Exited"].value_counts().rename({0: "Retained", 1: "Churned"}).reset_index()
    churn_counts.columns = ["Status", "Count"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=churn_counts["Status"],
                values=churn_counts["Count"],
                hole=0.58,
                marker=dict(colors=[COLOR_RETAINED, COLOR_CHURNED]),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont_size=13,
            )
        ]
    )
    fig.update_layout(
        title="<b>Customer Retention vs Churn Distribution</b>",
        title_font_size=16,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=30, l=20, r=20),
        height=350,
    )
    return fig


def plot_churn_rate_by_dimension(
    df: pd.DataFrame,
    dimension_col: str,
    title: str,
    x_label: Optional[str] = None,
    color: str = COLOR_PRIMARY,
) -> go.Figure:
    """Plot Churn Rate (%) bar chart across a categorical dimension."""
    if len(df) == 0 or dimension_col not in df.columns:
        return create_empty_figure()

    agg = (
        df.groupby(dimension_col, observed=False)
        .agg(Total=("Exited", "count"), Churned=("Exited", lambda x: (x == 1).sum()))
        .reset_index()
    )
    agg["Churn_Rate"] = (agg["Churned"] / agg["Total"] * 100.0).fillna(0.0).round(2)

    fig = go.Figure(
        data=[
            go.Bar(
                x=agg[dimension_col],
                y=agg["Churn_Rate"],
                text=agg.apply(lambda r: f"{r['Churn_Rate']:.1f}%<br>(n={r['Total']:,})", axis=1),
                textposition="outside",
                marker_color=color,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"Churn Rate: %{{y:.2f}}%<br>"
                    f"Total Customers: %{{customdata[0]:,}}<br>"
                    f"Churned: %{{customdata[1]:,}}<extra></extra>"
                ),
                customdata=agg[["Total", "Churned"]],
            )
        ]
    )

    overall_churn_rate = (df["Exited"] == 1).mean() * 100.0
    fig.add_hline(
        y=overall_churn_rate,
        line_dash="dash",
        line_color=COLOR_CHURNED,
        annotation_text=f"Benchmark ({overall_churn_rate:.1f}%)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        title_font_size=15,
        xaxis_title=x_label or dimension_col.replace("_", " "),
        yaxis_title="Observed Churn Rate (%)",
        yaxis=dict(range=[0, max(agg["Churn_Rate"].max() * 1.25, 10)]),
        margin=dict(t=50, b=40, l=50, r=30),
        height=380,
    )
    return fig


def plot_country_comparison_bar(df: pd.DataFrame) -> go.Figure:
    """Plot multi-metric country benchmark comparing customer volume and churn count."""
    if len(df) == 0:
        return create_empty_figure()

    agg = (
        df.groupby("Geography", observed=False)
        .agg(
            Total=("Exited", "count"),
            Churned=("Exited", lambda x: (x == 1).sum()),
            Retained=("Exited", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Retained",
            x=agg["Geography"],
            y=agg["Retained"],
            marker_color=COLOR_RETAINED,
            text=agg["Retained"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Churned",
            x=agg["Geography"],
            y=agg["Churned"],
            marker_color=COLOR_CHURNED,
            text=agg["Churned"],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="stack",
        title="<b>Customer Volume & Churn Count by Geography</b>",
        title_font_size=15,
        xaxis_title="Country",
        yaxis_title="Number of Customers",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=50, b=40, l=50, r=30),
        height=380,
    )
    return fig


def plot_heatmap_churn_matrix(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    title: str,
    y_label: str,
    x_label: str,
) -> go.Figure:
    """Generate 2D interactive Heatmap displaying Churn Rates (%) across cross-segments."""
    if len(df) == 0 or index_col not in df.columns or columns_col not in df.columns:
        return create_empty_figure()

    pivot_rates = df.pivot_table(
        index=index_col,
        columns=columns_col,
        values="Exited",
        aggfunc=lambda x: (x == 1).sum() / len(x) * 100.0 if len(x) > 0 else 0.0,
        observed=False,
    ).round(1)

    pivot_counts = df.pivot_table(
        index=index_col,
        columns=columns_col,
        values="Exited",
        aggfunc="count",
        observed=False,
    ).fillna(0).astype(int)

    text_matrix = []
    for r in pivot_rates.index:
        row_text = []
        for c in pivot_rates.columns:
            rate = pivot_rates.loc[r, c]
            count = pivot_counts.loc[r, c]
            row_text.append(f"{rate:.1f}%<br>(n={count:,})")
        text_matrix.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_rates.values,
            x=list(pivot_rates.columns.astype(str)),
            y=list(pivot_rates.index.astype(str)),
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale="Reds",
            colorbar=dict(title="Churn Rate %"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        title_font_size=15,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(t=50, b=40, l=80, r=40),
        height=400,
    )
    return fig


def plot_distribution_comparison(
    df: pd.DataFrame, feature_col: str, title: str, x_label: str
) -> go.Figure:
    """Plot comparative distribution box plots with overlaid jitter for churned vs retained."""
    if len(df) == 0 or feature_col not in df.columns:
        return create_empty_figure()

    fig = px.box(
        df,
        x="Churn_Label" if "Churn_Label" in df.columns else "Exited",
        y=feature_col,
        color="Churn_Label" if "Churn_Label" in df.columns else "Exited",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED, 0: COLOR_RETAINED, 1: COLOR_CHURNED},
        points="outliers",
        notched=True,
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        title_font_size=15,
        xaxis_title="Customer Status",
        yaxis_title=x_label,
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=30),
        height=380,
    )
    return fig


def plot_bubble_segment_risk(df: pd.DataFrame, segment_col: str, title: str) -> go.Figure:
    """Plot segment risk bubble chart: Size (Total Customers) vs X (Balance Exposure) vs Y (Churn Rate)."""
    if len(df) == 0 or segment_col not in df.columns:
        return create_empty_figure()

    agg = (
        df.groupby(segment_col, observed=False)
        .agg(
            Total=("Exited", "count"),
            Churned=("Exited", lambda x: (x == 1).sum()),
            Exposure=("Balance", lambda b: b[df.loc[b.index, "Exited"] == 1].sum()),
            Avg_Balance=("Balance", "mean"),
        )
        .reset_index()
    )
    agg["Churn_Rate"] = (agg["Churned"] / agg["Total"] * 100.0).round(2)

    fig = px.scatter(
        agg,
        x="Exposure",
        y="Churn_Rate",
        size="Total",
        color=segment_col,
        text=segment_col,
        hover_name=segment_col,
        size_max=45,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title=f"<b>{title}</b>",
        title_font_size=15,
        xaxis_title="Balance Exposure (€ Held by Churned Accounts)",
        yaxis_title="Observed Churn Rate (%)",
        margin=dict(t=50, b=40, l=60, r=30),
        height=400,
    )
    return fig
