"""KPI calculation engine for European Banking customer churn analysis.

Provides robust, dynamic calculations for portfolio-level, segment-level,
and high-value risk metrics with zero-division safety.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def calculate_overall_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate core portfolio-level KPIs from a customer dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (either full or filtered).

    Returns
    -------
    dict
        Comprehensive portfolio metrics.
    """
    total_customers = len(df)
    if total_customers == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "retained_customers": 0,
            "churn_rate": 0.0,
            "retention_rate": 0.0,
            "avg_credit_score": 0.0,
            "avg_age": 0.0,
            "avg_tenure": 0.0,
            "avg_balance": 0.0,
            "avg_salary": 0.0,
            "avg_num_products": 0.0,
            "active_member_rate": 0.0,
            "credit_card_rate": 0.0,
            "total_balance_exposure": 0.0,
            "total_portfolio_balance": 0.0,
        }

    churned = int((df["Exited"] == 1).sum())
    retained = int((df["Exited"] == 0).sum())
    churn_rate = (churned / total_customers) * 100.0
    retention_rate = (retained / total_customers) * 100.0

    avg_credit_score = float(df["CreditScore"].mean())
    avg_age = float(df["Age"].mean())
    avg_tenure = float(df["Tenure"].mean())
    avg_balance = float(df["Balance"].mean())
    avg_salary = float(df["EstimatedSalary"].mean())
    avg_num_products = float(df["NumOfProducts"].mean())

    active_members = int((df["IsActiveMember"] == 1).sum())
    active_member_rate = (active_members / total_customers) * 100.0

    credit_card_holders = int((df["HasCrCard"] == 1).sum())
    credit_card_rate = (credit_card_holders / total_customers) * 100.0

    total_portfolio_balance = float(df["Balance"].sum())
    churned_df = df[df["Exited"] == 1]
    total_balance_exposure = float(churned_df["Balance"].sum())

    return {
        "total_customers": total_customers,
        "churned_customers": churned,
        "retained_customers": retained,
        "churn_rate": round(churn_rate, 2),
        "retention_rate": round(retention_rate, 2),
        "avg_credit_score": round(avg_credit_score, 1),
        "avg_age": round(avg_age, 1),
        "avg_tenure": round(avg_tenure, 1),
        "avg_balance": round(avg_balance, 2),
        "avg_salary": round(avg_salary, 2),
        "avg_num_products": round(avg_num_products, 2),
        "active_member_rate": round(active_member_rate, 2),
        "credit_card_rate": round(credit_card_rate, 2),
        "total_portfolio_balance": round(total_portfolio_balance, 2),
        "total_balance_exposure": round(total_balance_exposure, 2),
    }


def calculate_high_value_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate high-value customer metrics and balance exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'Is_High_Value', 'Balance', 'Exited'.

    Returns
    -------
    dict
        High-value customer risk metrics.
    """
    total_customers = len(df)
    if total_customers == 0:
        return {
            "high_value_customer_count": 0,
            "high_value_churn_count": 0,
            "high_value_churn_rate": 0.0,
            "high_value_churn_ratio": 0.0,
            "high_value_balance_exposure": 0.0,
            "high_balance_churn_count": 0,
            "high_value_retention_rate": 0.0,
        }

    # Ensure Is_High_Value exists
    if "Is_High_Value" not in df.columns:
        from src.feature_engineering import calculate_balance_thresholds, calculate_salary_thresholds
        b_p33, b_p66 = calculate_balance_thresholds(df)
        s_p33, s_p66 = calculate_salary_thresholds(df)
        hv_mask = (df["Balance"] > b_p66) & (df["EstimatedSalary"] > s_p66)
    else:
        hv_mask = df["Is_High_Value"] == 1

    hv_df = df[hv_mask]
    hv_count = len(hv_df)
    hv_churn_df = hv_df[hv_df["Exited"] == 1]
    hv_churn_count = len(hv_churn_df)

    total_churn_count = int((df["Exited"] == 1).sum())

    hv_churn_rate = (hv_churn_count / hv_count * 100.0) if hv_count > 0 else 0.0
    hv_retention_rate = 100.0 - hv_churn_rate if hv_count > 0 else 0.0
    hv_churn_ratio = (hv_churn_count / total_churn_count * 100.0) if total_churn_count > 0 else 0.0
    hv_balance_exposure = float(hv_churn_df["Balance"].sum())

    # High balance churn count (> 66th percentile of non-zero balance)
    non_zero = df[df["Balance"] > 0]["Balance"]
    b_p66 = float(non_zero.quantile(0.66)) if len(non_zero) > 0 else 0.0
    high_balance_churn_count = int(((df["Balance"] > b_p66) & (df["Exited"] == 1)).sum())

    return {
        "high_value_customer_count": hv_count,
        "high_value_churn_count": hv_churn_count,
        "high_value_churn_rate": round(hv_churn_rate, 2),
        "high_value_retention_rate": round(hv_retention_rate, 2),
        "high_value_churn_ratio": round(hv_churn_ratio, 2),
        "high_value_balance_exposure": round(hv_balance_exposure, 2),
        "high_balance_churn_count": high_balance_churn_count,
    }


def calculate_segment_kpis(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """Calculate detailed customer count, churn count, churn rate, and financials by segment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    segment_column : str
        Column name to group by.

    Returns
    -------
    pd.DataFrame
        Aggregated summary table.
    """
    if len(df) == 0 or segment_column not in df.columns:
        return pd.DataFrame(
            columns=[
                segment_column,
                "Total_Customers",
                "Retained_Customers",
                "Churned_Customers",
                "Churn_Rate_Pct",
                "Retention_Rate_Pct",
                "Avg_Balance",
                "Total_Balance_Exposure",
                "Avg_CreditScore",
                "Avg_Age",
                "Active_Member_Rate_Pct",
            ]
        )

    grouped = df.groupby(segment_column, observed=False)

    summary = grouped.agg(
        Total_Customers=("Exited", "count"),
        Churned_Customers=("Exited", lambda x: (x == 1).sum()),
        Retained_Customers=("Exited", lambda x: (x == 0).sum()),
        Avg_Balance=("Balance", "mean"),
        Total_Balance_Exposure=("Balance", lambda b: b[df.loc[b.index, "Exited"] == 1].sum()),
        Avg_CreditScore=("CreditScore", "mean"),
        Avg_Age=("Age", "mean"),
        Active_Member_Rate_Pct=("IsActiveMember", lambda m: (m == 1).mean() * 100.0),
    ).reset_index()

    summary["Churn_Rate_Pct"] = (
        summary["Churned_Customers"] / summary["Total_Customers"] * 100.0
    ).fillna(0.0).round(2)
    summary["Retention_Rate_Pct"] = (
        summary["Retained_Customers"] / summary["Total_Customers"] * 100.0
    ).fillna(0.0).round(2)
    summary["Avg_Balance"] = summary["Avg_Balance"].round(2)
    summary["Total_Balance_Exposure"] = summary["Total_Balance_Exposure"].round(2)
    summary["Avg_CreditScore"] = summary["Avg_CreditScore"].round(1)
    summary["Avg_Age"] = summary["Avg_Age"].round(1)
    summary["Active_Member_Rate_Pct"] = summary["Active_Member_Rate_Pct"].round(2)

    return summary
