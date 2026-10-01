"""Unit tests for data cleaning and validation routines."""
import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.data_cleaning import clean_data, validate_data, EXPECTED_COLUMNS, EXPECTED_GEOGRAPHIES, EXPECTED_GENDERS


@pytest.fixture
def raw_data():
    return load_raw_data()


def test_raw_data_schema(raw_data):
    """Verify raw dataset contains all 14 expected columns and 10,000 rows."""
    assert raw_data.shape == (10000, 14)
    for col in EXPECTED_COLUMNS:
        assert col in raw_data.columns


def test_data_validation_summary(raw_data):
    """Verify data validation reports no missing values or duplicate customer IDs."""
    summary = validate_data(raw_data)
    assert summary["is_clean"] is True
    assert summary["total_missing"] == 0
    assert summary["duplicate_rows"] == 0
    assert summary["duplicate_customer_ids"] == 0


def test_categorical_domains(raw_data):
    """Verify Geography and Gender values match expected banking domains."""
    geos = set(raw_data["Geography"].unique())
    genders = set(raw_data["Gender"].unique())
    assert geos == EXPECTED_GEOGRAPHIES
    assert genders == EXPECTED_GENDERS


def test_binary_columns(raw_data):
    """Verify binary columns contain only {0, 1}."""
    for col in ["Exited", "HasCrCard", "IsActiveMember"]:
        unique_vals = set(raw_data[col].unique())
        assert unique_vals.issubset({0, 1})


def test_clean_data_transformation(raw_data):
    """Verify data cleaning drops Surname and preserves CustomerId and Year."""
    cleaned = clean_data(raw_data, drop_surname=True)
    assert "Surname" not in cleaned.columns
    assert "CustomerId" in cleaned.columns
    assert "Year" in cleaned.columns
    assert len(cleaned) == 10000
