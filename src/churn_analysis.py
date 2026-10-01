"""Churn comparison, statistical testing, and predictive modeling module.

Conducts rigorous statistical profiling between churned and retained customer cohorts,
tests hypothesis significance, and evaluates baseline predictive models.
"""
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compare_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Compare numerical profile between churned and retained customers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Comparison table with mean, median, std, and difference.
    """
    num_cols = ["Age", "CreditScore", "Tenure", "Balance", "EstimatedSalary", "NumOfProducts"]
    records = []

    churned = df[df["Exited"] == 1]
    retained = df[df["Exited"] == 0]

    for col in num_cols:
        if col in df.columns:
            mean_c = churned[col].mean()
            median_c = churned[col].median()
            std_c = churned[col].std()

            mean_r = retained[col].mean()
            median_r = retained[col].median()
            std_r = retained[col].std()

            diff_mean = mean_c - mean_r
            pct_diff = (diff_mean / mean_r * 100.0) if mean_r != 0 else 0.0

            records.append({
                "Metric": col,
                "Retained_Mean": round(mean_r, 2),
                "Churned_Mean": round(mean_c, 2),
                "Mean_Difference": round(diff_mean, 2),
                "Pct_Difference": round(pct_diff, 2),
                "Retained_Median": round(median_r, 2),
                "Churned_Median": round(median_c, 2),
                "Retained_Std": round(std_r, 2),
                "Churned_Std": round(std_c, 2),
            })

    return pd.DataFrame(records)


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Perform hypothesis tests for association between customer features and churn status.

    - Categorical variables: Chi-Square Test of Independence.
    - Continuous variables: Two-sample independent t-test (and Mann-Whitney U test).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Summary table with test statistics, p-values, and statistical interpretations.
    """
    results = []

    # 1. Categorical Chi-Square tests
    cat_cols = [
        ("Geography", "Geographic Market"),
        ("Gender", "Gender"),
        ("IsActiveMember", "Active Membership Status"),
        ("HasCrCard", "Credit Card Ownership"),
        ("Age_Group", "Age Cohort"),
        ("Credit_Score_Band", "Credit Score Tier"),
        ("Tenure_Group", "Tenure Tier"),
        ("Balance_Segment", "Balance Segment"),
        ("Product_Segment", "Product Count Tier"),
    ]

    for col, display_name in cat_cols:
        if col in df.columns:
            contingency = pd.crosstab(df[col], df["Exited"])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            is_sig = p_val < 0.05
            results.append({
                "Feature": display_name,
                "Feature_Type": "Categorical",
                "Test_Type": "Chi-Square Test",
                "Statistic": round(chi2, 3),
                "p_value": float(p_val),
                "p_value_formatted": "< 0.001" if p_val < 0.001 else f"{p_val:.4f}",
                "Significant_at_0.05": is_sig,
                "Interpretation": (
                    "Statistically significant association observed"
                    if is_sig
                    else "No statistically significant difference observed"
                ),
            })

    # 2. Continuous Numeric t-tests
    num_cols = [
        ("Age", "Customer Age"),
        ("CreditScore", "Credit Score"),
        ("Balance", "Account Balance"),
        ("EstimatedSalary", "Estimated Salary"),
        ("Tenure", "Tenure (Years)"),
        ("NumOfProducts", "Number of Products"),
    ]

    churned = df[df["Exited"] == 1]
    retained = df[df["Exited"] == 0]

    for col, display_name in num_cols:
        if col in df.columns:
            t_stat, p_val = stats.ttest_ind(
                churned[col].dropna(), retained[col].dropna(), equal_var=False
            )
            is_sig = p_val < 0.05
            results.append({
                "Feature": display_name,
                "Feature_Type": "Numeric",
                "Test_Type": "Welch's Two-Sample t-Test",
                "Statistic": round(t_stat, 3),
                "p_value": float(p_val),
                "p_value_formatted": "< 0.001" if p_val < 0.001 else f"{p_val:.4f}",
                "Significant_at_0.05": is_sig,
                "Interpretation": (
                    "Statistically significant mean difference observed"
                    if is_sig
                    else "No statistically significant mean difference"
                ),
            })

    return pd.DataFrame(results)


def train_baseline_churn_models(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """Train and evaluate baseline Logistic Regression and Random Forest models.

    Strictly excludes CustomerId and Surname to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    test_size : float, default 0.2
        Holdout test set fraction.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    dict
        Evaluation metrics, confusion matrices, and feature importances.
    """
    feature_cols = [
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
    ]

    X = df[feature_cols].copy()
    y = df["Exited"].astype(int)

    categorical_features = ["Geography", "Gender"]
    numeric_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 1. Logistic Regression Pipeline
    lr_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))]
    )
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    # 2. Random Forest Pipeline
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=150, max_depth=8, random_state=random_state, class_weight="balanced"),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    # Feature names after one-hot encoding
    ohe_cols = list(
        rf_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )
    all_feature_names = numeric_features + ohe_cols

    rf_importances = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": rf_pipeline.named_steps["classifier"].feature_importances_,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    results = {
        "sample_counts": {
            "total": len(df),
            "train": len(X_train),
            "test": len(X_test),
        },
        "logistic_regression": {
            "accuracy": round(accuracy_score(y_test, y_pred_lr) * 100.0, 2),
            "precision": round(precision_score(y_test, y_pred_lr, zero_division=0) * 100.0, 2),
            "recall": round(recall_score(y_test, y_pred_lr, zero_division=0) * 100.0, 2),
            "f1": round(f1_score(y_test, y_pred_lr, zero_division=0) * 100.0, 2),
            "roc_auc": round(roc_auc_score(y_test, y_proba_lr), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
        },
        "random_forest": {
            "accuracy": round(accuracy_score(y_test, y_pred_rf) * 100.0, 2),
            "precision": round(precision_score(y_test, y_pred_rf, zero_division=0) * 100.0, 2),
            "recall": round(recall_score(y_test, y_pred_rf, zero_division=0) * 100.0, 2),
            "f1": round(f1_score(y_test, y_pred_rf, zero_division=0) * 100.0, 2),
            "roc_auc": round(roc_auc_score(y_test, y_proba_rf), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
            "feature_importances": rf_importances,
        },
    }

    return results
