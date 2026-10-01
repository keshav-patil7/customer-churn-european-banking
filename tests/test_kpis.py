"""Unit tests for portfolio and high-value KPI calculations."""
import pytest
import pandas as pd
from src.data_loader import load_cleaned_data
from src.kpi_calculations import (
    calculate_overall_kpis,
    calculate_high_value_kpis,
    calculate_segment_kpis,
)


@pytest.fixture
def dataset():
    return load_cleaned_data()


def test_overall_kpis_validity(dataset):
    """Verify overall KPIs satisfy mathematical bounds."""
    kpis = calculate_overall_kpis(dataset)
    assert kpis["total_customers"] == 10000
    assert kpis["churned_customers"] == 2037
    assert kpis["retained_customers"] == 7963
    assert 0.0 <= kpis["churn_rate"] <= 100.0
    assert 0.0 <= kpis["retention_rate"] <= 100.0
    assert abs((kpis["churn_rate"] + kpis["retention_rate"]) - 100.0) < 0.1
    assert kpis["total_balance_exposure"] > 0
    assert kpis["total_balance_exposure"] < kpis["total_portfolio_balance"]


def test_high_value_kpis_validity(dataset):
    """Verify high-value KPIs satisfy consistency constraints."""
    hv = calculate_high_value_kpis(dataset)
    assert hv["high_value_customer_count"] > 0
    assert hv["high_value_churn_count"] <= hv["high_value_customer_count"]
    assert 0.0 <= hv["high_value_churn_rate"] <= 100.0
    assert 0.0 <= hv["high_value_churn_ratio"] <= 100.0
    assert hv["high_value_balance_exposure"] > 0


def test_empty_dataframe_kpis():
    """Verify calculations return safe defaults without division by zero on empty data."""
    empty_df = pd.DataFrame(columns=["Exited", "Balance", "CreditScore", "Age", "Tenure", "EstimatedSalary", "NumOfProducts", "IsActiveMember", "HasCrCard"])
    kpis = calculate_overall_kpis(empty_df)
    assert kpis["total_customers"] == 0
    assert kpis["churn_rate"] == 0.0
    assert kpis["retention_rate"] == 0.0

    hv = calculate_high_value_kpis(empty_df)
    assert hv["high_value_customer_count"] == 0
    assert hv["high_value_churn_rate"] == 0.0


def test_segment_kpis_summation(dataset):
    """Verify segment KPI sums equal overall dataset totals."""
    geo_kpis = calculate_segment_kpis(dataset, "Geography")
    assert geo_kpis["Total_Customers"].sum() == 10000
    assert geo_kpis["Churned_Customers"].sum() == 2037
    assert geo_kpis["Retained_Customers"].sum() == 7963
