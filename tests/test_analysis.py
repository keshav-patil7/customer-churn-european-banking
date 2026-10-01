"""Unit tests for churn analysis, segmentation tables, and statistical testing."""
import pytest
import pandas as pd
from src.data_loader import load_cleaned_data
from src.segmentation import analyze_univariate_segment, analyze_multivariate_cross_segment
from src.churn_analysis import compare_cohorts, run_statistical_tests, train_baseline_churn_models


@pytest.fixture
def dataset():
    return load_cleaned_data()


def test_univariate_segment_analysis(dataset):
    """Verify univariate segmentation aggregations."""
    age_seg = analyze_univariate_segment(dataset, "Age_Group")
    assert not age_seg.empty
    assert age_seg["Customer_Count"].sum() == 10000
    assert age_seg["Churn_Count"].sum() == 2037
    assert (age_seg["Churn_Rate_Pct"] >= 0).all()
    assert (age_seg["Churn_Rate_Pct"] <= 100).all()


def test_multivariate_cross_tabulation(dataset):
    """Verify 2D cross-tabulation sums and structure."""
    cross = analyze_multivariate_cross_segment(dataset, "Geography", "Age_Group")
    assert not cross.empty
    assert cross["Customer_Count"].sum() == 10000
    assert cross["Churn_Count"].sum() == 2037


def test_cohort_comparison(dataset):
    """Verify cohort differences calculate without error."""
    diff = compare_cohorts(dataset)
    assert not diff.empty
    assert "Metric" in diff.columns
    assert "Mean_Difference" in diff.columns


def test_statistical_hypothesis_tests(dataset):
    """Verify statistical testing executes and returns valid p-values."""
    stat_df = run_statistical_tests(dataset)
    assert not stat_df.empty
    assert "p_value" in stat_df.columns
    assert ((stat_df["p_value"] >= 0.0) & (stat_df["p_value"] <= 1.0)).all()


def test_baseline_churn_models(dataset):
    """Verify baseline ML pipeline runs and reports reasonable classification metrics."""
    res = train_baseline_churn_models(dataset, test_size=0.2, random_state=42)
    assert "logistic_regression" in res
    assert "random_forest" in res
    assert 0.5 <= res["logistic_regression"]["roc_auc"] <= 1.0
    assert 0.5 <= res["random_forest"]["roc_auc"] <= 1.0
