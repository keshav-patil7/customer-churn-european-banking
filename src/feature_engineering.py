"""Feature engineering and customer segmentation module.

Computes empirical thresholds dynamically from data and creates derived segmentation
features for banking behavioral and financial profile analysis.
"""
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np


def calculate_balance_thresholds(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate the 33rd and 66th percentiles of non-zero account balances.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'Balance' column.

    Returns
    -------
    Tuple[float, float]
        (p33, p66) of non-zero balances.
    """
    non_zero = df[df["Balance"] > 0]["Balance"]
    if len(non_zero) == 0:
        return 0.0, 0.0
    p33 = float(non_zero.quantile(0.33))
    p66 = float(non_zero.quantile(0.66))
    return p33, p66


def calculate_salary_thresholds(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate the 33rd and 66th percentiles of EstimatedSalary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'EstimatedSalary' column.

    Returns
    -------
    Tuple[float, float]
        (p33, p66) of estimated salaries.
    """
    p33 = float(df["EstimatedSalary"].quantile(0.33))
    p66 = float(df["EstimatedSalary"].quantile(0.66))
    return p33, p66


def get_segmentation_thresholds(df: pd.DataFrame) -> Dict[str, Any]:
    """Return all dynamically computed thresholds used across the project."""
    b_p33, b_p66 = calculate_balance_thresholds(df)
    s_p33, s_p66 = calculate_salary_thresholds(df)

    return {
        "balance_p33_nonzero": b_p33,
        "balance_p66_nonzero": b_p66,
        "salary_p33": s_p33,
        "salary_p66": s_p66,
        "credit_score_low_cutoff": 580,
        "credit_score_high_cutoff": 670,
        "age_bins": ["<30", "30-45", "46-60", "60+"],
        "tenure_bins": ["New (0-2)", "Mid-term (3-7)", "Long-term (8-10)"],
        "product_bins": ["1 Product", "2 Products", "3+ Products"],
    }


def categorize_age(age: int) -> str:
    """Categorize customer age into mutually exclusive cohorts."""
    if age < 30:
        return "<30"
    elif 30 <= age <= 45:
        return "30-45"
    elif 46 <= age <= 60:
        return "46-60"
    else:
        return "60+"


def categorize_credit_score(score: int, low_cutoff: int = 580, high_cutoff: int = 670) -> str:
    """Categorize credit score using banking industry risk bands."""
    if score < low_cutoff:
        return "Low"
    elif score < high_cutoff:
        return "Medium"
    else:
        return "High"


def categorize_tenure(tenure: int) -> str:
    """Categorize tenure into relationship maturity stages."""
    if tenure <= 2:
        return "New (0-2)"
    elif tenure <= 7:
        return "Mid-term (3-7)"
    else:
        return "Long-term (8-10)"


def categorize_balance(balance: float, p33: float, p66: float) -> str:
    """Categorize account balance isolating zero-balance accounts and percentile tiers."""
    if balance == 0:
        return "Zero Balance"
    elif balance <= p33:
        return "Low Balance"
    elif balance <= p66:
        return "Medium Balance"
    else:
        return "High Balance"


def categorize_salary(salary: float, p33: float, p66: float) -> str:
    """Categorize estimated salary using dynamic dataset tertiles."""
    if salary <= p33:
        return "Low Salary"
    elif salary <= p66:
        return "Medium Salary"
    else:
        return "High Salary"


def categorize_products(num_products: int) -> str:
    """Categorize product count with 3+ grouped for statistical stability."""
    if num_products == 1:
        return "1 Product"
    elif num_products == 2:
        return "2 Products"
    else:
        return "3+ Products"


def add_features(
    df: pd.DataFrame,
    balance_thresholds: Optional[Tuple[float, float]] = None,
    salary_thresholds: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Engineer all derived analytical segments onto the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input dataframe.
    balance_thresholds : Optional[Tuple[float, float]]
        Precomputed (p33, p66) for non-zero balance. If None, computed dynamically.
    salary_thresholds : Optional[Tuple[float, float]]
        Precomputed (p33, p66) for estimated salary. If None, computed dynamically.

    Returns
    -------
    pd.DataFrame
        Dataframe with added segmentation columns.
    """
    featured_df = df.copy()

    # Dynamic threshold extraction
    if balance_thresholds is None:
        b_p33, b_p66 = calculate_balance_thresholds(featured_df)
    else:
        b_p33, b_p66 = balance_thresholds

    if salary_thresholds is None:
        s_p33, s_p66 = calculate_salary_thresholds(featured_df)
    else:
        s_p33, s_p66 = salary_thresholds

    # 1. Age Group
    featured_df["Age_Group"] = featured_df["Age"].apply(categorize_age)
    age_cat_order = ["<30", "30-45", "46-60", "60+"]
    featured_df["Age_Group"] = pd.Categorical(featured_df["Age_Group"], categories=age_cat_order, ordered=True)

    # 2. Credit Score Band
    featured_df["Credit_Score_Band"] = featured_df["CreditScore"].apply(categorize_credit_score)
    cs_cat_order = ["Low", "Medium", "High"]
    featured_df["Credit_Score_Band"] = pd.Categorical(
        featured_df["Credit_Score_Band"], categories=cs_cat_order, ordered=True
    )

    # 3. Tenure Group
    featured_df["Tenure_Group"] = featured_df["Tenure"].apply(categorize_tenure)
    tenure_cat_order = ["New (0-2)", "Mid-term (3-7)", "Long-term (8-10)"]
    featured_df["Tenure_Group"] = pd.Categorical(
        featured_df["Tenure_Group"], categories=tenure_cat_order, ordered=True
    )

    # 4. Balance Segment
    featured_df["Balance_Segment"] = featured_df["Balance"].apply(lambda b: categorize_balance(b, b_p33, b_p66))
    balance_cat_order = ["Zero Balance", "Low Balance", "Medium Balance", "High Balance"]
    featured_df["Balance_Segment"] = pd.Categorical(
        featured_df["Balance_Segment"], categories=balance_cat_order, ordered=True
    )

    # 5. Salary Segment
    featured_df["Salary_Segment"] = featured_df["EstimatedSalary"].apply(lambda s: categorize_salary(s, s_p33, s_p66))
    salary_cat_order = ["Low Salary", "Medium Salary", "High Salary"]
    featured_df["Salary_Segment"] = pd.Categorical(
        featured_df["Salary_Segment"], categories=salary_cat_order, ordered=True
    )

    # 6. Product Segment
    featured_df["Product_Segment"] = featured_df["NumOfProducts"].apply(categorize_products)
    product_cat_order = ["1 Product", "2 Products", "3+ Products"]
    featured_df["Product_Segment"] = pd.Categorical(
        featured_df["Product_Segment"], categories=product_cat_order, ordered=True
    )

    # 7. High-Value Customer Flag: Top non-zero balance tier AND Top salary tier
    featured_df["Is_High_Value"] = (
        (featured_df["Balance"] > b_p66) & (featured_df["EstimatedSalary"] > s_p66)
    ).astype(int)

    # 8. Human-readable Status Labels for charts
    featured_df["Churn_Label"] = featured_df["Exited"].map({0: "Retained", 1: "Churned"})
    featured_df["Activity_Label"] = featured_df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    featured_df["CreditCard_Label"] = featured_df["HasCrCard"].map({0: "No Credit Card", 1: "Has Credit Card"})

    return featured_df
