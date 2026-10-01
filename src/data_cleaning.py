"""Data cleaning and validation module for European Banking dataset.

Performs thorough quality checks, type enforcement, duplicate detection,
and sanitization while maintaining auditability.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


EXPECTED_COLUMNS = [
    "Year",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]

EXPECTED_GEOGRAPHIES = {"France", "Spain", "Germany"}
EXPECTED_GENDERS = {"Male", "Female"}


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality inspection and return a diagnostic summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.

    Returns
    -------
    dict
        Dictionary containing quality metrics, issues found, and column statistics.
    """
    total_rows, total_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    unique_customer_ids = int(df["CustomerId"].nunique()) if "CustomerId" in df.columns else 0
    duplicate_customer_ids = total_rows - unique_customer_ids

    issues = []
    if total_missing > 0:
        issues.append(f"Found {total_missing} missing values across columns.")
    if duplicate_rows > 0:
        issues.append(f"Found {duplicate_rows} duplicate rows.")
    if duplicate_customer_ids > 0:
        issues.append(f"Found {duplicate_customer_ids} duplicate CustomerId records.")

    # Check binary constraints
    for bin_col in ["Exited", "HasCrCard", "IsActiveMember"]:
        if bin_col in df.columns:
            invalid_bin = df[~df[bin_col].isin([0, 1])]
            if not invalid_bin.empty:
                issues.append(f"Column '{bin_col}' contains {len(invalid_bin)} non-binary values.")

    # Check categorical domains
    if "Geography" in df.columns:
        invalid_geo = set(df["Geography"].dropna().unique()) - EXPECTED_GEOGRAPHIES
        if invalid_geo:
            issues.append(f"Unexpected Geography values: {invalid_geo}")

    if "Gender" in df.columns:
        invalid_gender = set(df["Gender"].dropna().unique()) - EXPECTED_GENDERS
        if invalid_gender:
            issues.append(f"Unexpected Gender values: {invalid_gender}")

    # Check numerical validity
    if "Age" in df.columns:
        invalid_age = df[(df["Age"] < 18) | (df["Age"] > 100)]
        if not invalid_age.empty:
            issues.append(f"Found {len(invalid_age)} records with Age out of normal bounds (18-100).")

    if "CreditScore" in df.columns:
        invalid_cs = df[(df["CreditScore"] < 300) | (df["CreditScore"] > 850)]
        if not invalid_cs.empty:
            issues.append(f"Found {len(invalid_cs)} records with CreditScore out of bounds (300-850).")

    if "Balance" in df.columns:
        neg_bal = df[df["Balance"] < 0]
        if not neg_bal.empty:
            issues.append(f"Found {len(neg_bal)} records with negative Balance.")

    if "EstimatedSalary" in df.columns:
        neg_sal = df[df["EstimatedSalary"] < 0]
        if not neg_sal.empty:
            issues.append(f"Found {len(neg_sal)} records with negative EstimatedSalary.")

    if "NumOfProducts" in df.columns:
        invalid_prod = df[df["NumOfProducts"] < 1]
        if not invalid_prod.empty:
            issues.append(f"Found {len(invalid_prod)} records with NumOfProducts < 1.")

    summary = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_counts": missing_counts,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "unique_customer_ids": unique_customer_ids,
        "duplicate_customer_ids": duplicate_customer_ids,
        "geography_distribution": df["Geography"].value_counts().to_dict() if "Geography" in df.columns else {},
        "gender_distribution": df["Gender"].value_counts().to_dict() if "Gender" in df.columns else {},
        "target_distribution": df["Exited"].value_counts().to_dict() if "Exited" in df.columns else {},
        "issues_found": issues,
        "is_clean": len(issues) == 0,
    }
    return summary


def clean_data(df: pd.DataFrame, drop_surname: bool = True) -> pd.DataFrame:
    """Clean and sanitize the raw dataset.

    - Removes exact duplicate rows.
    - Strips whitespace from string columns.
    - Ensures correct datatypes.
    - Drops customer surname from analytical sets for privacy (if requested).
    - Preserves Year and CustomerId.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.
    drop_surname : bool, default True
        Whether to exclude the Surname column for analytical data privacy.

    Returns
    -------
    pd.DataFrame
        Sanitized dataframe.
    """
    cleaned_df = df.copy()

    # 1. Remove duplicate rows if any
    cleaned_df = cleaned_df.drop_duplicates()

    # 2. Standardize string columns
    str_cols = cleaned_df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # 3. Enforce types
    type_conversions = {
        "Year": "int64",
        "CustomerId": "int64",
        "CreditScore": "int64",
        "Age": "int64",
        "Tenure": "int64",
        "NumOfProducts": "int64",
        "HasCrCard": "int64",
        "IsActiveMember": "int64",
        "Exited": "int64",
        "Balance": "float64",
        "EstimatedSalary": "float64",
    }
    for col, dtype in type_conversions.items():
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(dtype)

    # 4. Remove Surname for privacy if requested
    if drop_surname and "Surname" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["Surname"])

    return cleaned_df
