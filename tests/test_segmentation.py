"""Unit tests for feature engineering and customer segmentation."""
import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.data_cleaning import clean_data
from src.feature_engineering import (
    add_features,
    calculate_balance_thresholds,
    calculate_salary_thresholds,
    get_segmentation_thresholds,
)


@pytest.fixture
def processed_data():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    return add_features(cleaned)


def test_balance_thresholds_calculation(processed_data):
    """Verify non-zero balance percentiles are strictly positive and correctly ordered."""
    p33, p66 = calculate_balance_thresholds(processed_data)
    assert p33 > 0
    assert p66 > p33
    # Non-zero balance p33 is ~106k, p66 is ~131k
    assert 100000 < p33 < 120000
    assert 120000 < p66 < 140000


def test_salary_thresholds_calculation(processed_data):
    """Verify estimated salary percentiles are strictly positive and ordered."""
    p33, p66 = calculate_salary_thresholds(processed_data)
    assert p33 > 0
    assert p66 > p33
    assert 50000 < p33 < 80000
    assert 110000 < p66 < 150000


def test_segment_completeness(processed_data):
    """Verify all segmentation columns are non-null and contain expected categories."""
    assert processed_data["Age_Group"].isnull().sum() == 0
    assert set(processed_data["Age_Group"].unique()) == {"<30", "30-45", "46-60", "60+"}

    assert processed_data["Credit_Score_Band"].isnull().sum() == 0
    assert set(processed_data["Credit_Score_Band"].unique()) == {"Low", "Medium", "High"}

    assert processed_data["Tenure_Group"].isnull().sum() == 0
    assert set(processed_data["Tenure_Group"].unique()) == {"New (0-2)", "Mid-term (3-7)", "Long-term (8-10)"}

    assert processed_data["Balance_Segment"].isnull().sum() == 0
    assert set(processed_data["Balance_Segment"].unique()) == {
        "Zero Balance",
        "Low Balance",
        "Medium Balance",
        "High Balance",
    }

    assert processed_data["Salary_Segment"].isnull().sum() == 0
    assert set(processed_data["Salary_Segment"].unique()) == {"Low Salary", "Medium Salary", "High Salary"}

    assert processed_data["Product_Segment"].isnull().sum() == 0
    assert set(processed_data["Product_Segment"].unique()) == {"1 Product", "2 Products", "3+ Products"}


def test_high_value_customer_definition(processed_data):
    """Verify high-value flag correctly identifies top financial tier."""
    hv_customers = processed_data[processed_data["Is_High_Value"] == 1]
    assert len(hv_customers) > 0
    assert len(hv_customers) < len(processed_data)
    # Every high-value customer must have non-zero balance and high salary
    assert (hv_customers["Balance"] > 0).all()
    assert (hv_customers["EstimatedSalary"] > 100000).all()
