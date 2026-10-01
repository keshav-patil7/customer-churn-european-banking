"""Multi-dimensional segmentation and cross-tabulation engine.

Computes 1D and 2D customer segment profiles, identifying high-risk combinations
and flagging small sample sizes for robust statistical governance.
"""
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


SMALL_SAMPLE_THRESHOLD = 50


def analyze_univariate_segment(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Compute univariate segment metrics including sample size flags.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    segment_col : str
        Column name to analyze.

    Returns
    -------
    pd.DataFrame
        Segment summary dataframe.
    """
    if len(df) == 0 or segment_col not in df.columns:
        return pd.DataFrame()

    agg_df = (
        df.groupby(segment_col, observed=False)
        .agg(
            Customer_Count=("Exited", "count"),
            Churn_Count=("Exited", lambda x: (x == 1).sum()),
            Retained_Count=("Exited", lambda x: (x == 0).sum()),
            Avg_Balance=("Balance", "mean"),
            Avg_Salary=("EstimatedSalary", "mean"),
            Avg_Age=("Age", "mean"),
            Avg_CreditScore=("CreditScore", "mean"),
            Active_Rate=("IsActiveMember", lambda x: (x == 1).mean() * 100.0),
            Balance_Exposure=("Balance", lambda b: b[df.loc[b.index, "Exited"] == 1].sum()),
        )
        .reset_index()
    )

    agg_df["Churn_Rate_Pct"] = (agg_df["Churn_Count"] / agg_df["Customer_Count"] * 100.0).fillna(0.0).round(2)
    agg_df["Retention_Rate_Pct"] = (agg_df["Retained_Count"] / agg_df["Customer_Count"] * 100.0).fillna(0.0).round(2)
    agg_df["Population_Share_Pct"] = (agg_df["Customer_Count"] / len(df) * 100.0).round(2)
    agg_df["Small_Sample_Flag"] = agg_df["Customer_Count"] < SMALL_SAMPLE_THRESHOLD

    # Round financials
    agg_df["Avg_Balance"] = agg_df["Avg_Balance"].round(2)
    agg_df["Avg_Salary"] = agg_df["Avg_Salary"].round(2)
    agg_df["Avg_Age"] = agg_df["Avg_Age"].round(1)
    agg_df["Avg_CreditScore"] = agg_df["Avg_CreditScore"].round(1)
    agg_df["Active_Rate"] = agg_df["Active_Rate"].round(2)
    agg_df["Balance_Exposure"] = agg_df["Balance_Exposure"].round(2)

    return agg_df


def analyze_multivariate_cross_segment(
    df: pd.DataFrame, dim1: str, dim2: str
) -> pd.DataFrame:
    """Compute 2-dimensional cross-tabulation with churn rates and sample size indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    dim1 : str
        First segmentation dimension (row axis).
    dim2 : str
        Second segmentation dimension (column axis).

    Returns
    -------
    pd.DataFrame
        Cross-tabulation summary table.
    """
    if len(df) == 0 or dim1 not in df.columns or dim2 not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby([dim1, dim2], observed=False)
        .agg(
            Customer_Count=("Exited", "count"),
            Churn_Count=("Exited", lambda x: (x == 1).sum()),
            Balance_Exposure=("Balance", lambda b: b[df.loc[b.index, "Exited"] == 1].sum()),
        )
        .reset_index()
    )

    grouped["Churn_Rate_Pct"] = (
        grouped["Churn_Count"] / grouped["Customer_Count"] * 100.0
    ).fillna(0.0).round(2)
    grouped["Small_Sample_Flag"] = grouped["Customer_Count"] < SMALL_SAMPLE_THRESHOLD

    return grouped


def generate_churn_pivot_table(
    df: pd.DataFrame, index_col: str, columns_col: str, metric: str = "churn_rate"
) -> pd.DataFrame:
    """Generate pivot table for heatmaps and cross-tab visualizers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    index_col : str
        Row dimension.
    columns_col : str
        Column dimension.
    metric : str, default 'churn_rate'
        Metric to display in cells ('churn_rate', 'customer_count', 'balance_exposure').

    Returns
    -------
    pd.DataFrame
        Pivot table.
    """
    if len(df) == 0 or index_col not in df.columns or columns_col not in df.columns:
        return pd.DataFrame()

    if metric == "churn_rate":
        pivot = df.pivot_table(
            index=index_col,
            columns=columns_col,
            values="Exited",
            aggfunc=lambda x: (x == 1).sum() / len(x) * 100.0 if len(x) > 0 else 0.0,
            observed=False,
        )
    elif metric == "customer_count":
        pivot = df.pivot_table(
            index=index_col,
            columns=columns_col,
            values="Exited",
            aggfunc="count",
            observed=False,
        )
    elif metric == "balance_exposure":
        pivot = df.pivot_table(
            index=index_col,
            columns=columns_col,
            values="Balance",
            aggfunc=lambda b: b[df.loc[b.index, "Exited"] == 1].sum(),
            observed=False,
        )
    else:
        raise ValueError(f"Unsupported pivot metric '{metric}'.")

    return pivot.fillna(0.0).round(2)
