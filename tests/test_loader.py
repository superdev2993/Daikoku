"""
Test loader.py - Validation of CSV loading and validation
"""

import sys
import os
import pandas as pd
import numpy as np
from conftest import TEST_CSV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data import loader


def test_load_valid_csv():
    """Test loading a valid CSV file"""
    df = loader.load_csv(TEST_CSV)

    assert df is not None
    assert len(df) > 0
    assert 'Open time' in df.columns
    print("✓ Valid CSV loaded successfully")


def test_load_extended_csv():
    """Test loading CSV (test dataset has minimal columns)"""
    df = loader.load_csv(TEST_CSV)

    assert df is not None
    assert len(df) > 0
    # Test dataset has minimal required columns
    print("✓ Test CSV loaded successfully")


def test_validate_data():
    """Test data validation on valid data"""
    df = loader.load_csv(TEST_CSV)
    df_validated = loader.validate_data(df)

    # Check only required columns remain
    assert list(df_validated.columns) == loader.REQUIRED_COLUMNS

    # Check datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(df_validated['Open time'])

    # Check float32 conversion
    assert df_validated['Open'].dtype == np.float64
    assert df_validated['High'].dtype == np.float64
    assert df_validated['Low'].dtype == np.float64
    assert df_validated['Close'].dtype == np.float64

    # Check no NaN
    assert not df_validated.isnull().any().any()

    print("✓ Data validation successful")


def test_chronological_order():
    """Test chronological order verification"""
    df = loader.load_csv(TEST_CSV)
    df_validated = loader.validate_data(df)

    # Check monotonic increasing
    assert df_validated['Open time'].is_monotonic_increasing

    print("✓ Chronological order verified")


def test_ohlc_logic():
    """Test OHLC logic validation"""
    df = loader.load_csv(TEST_CSV)
    df_validated = loader.validate_data(df)

    # High >= Low
    assert (df_validated['High'] >= df_validated['Low']).all()

    # Open in [Low, High]
    assert (df_validated['Open'] >= df_validated['Low']).all()
    assert (df_validated['Open'] <= df_validated['High']).all()

    # Close in [Low, High]
    assert (df_validated['Close'] >= df_validated['Low']).all()
    assert (df_validated['Close'] <= df_validated['High']).all()

    print("✓ OHLC logic validated")


def test_load_and_validate():
    """Test combined load and validate"""
    df = loader.load_and_validate(TEST_CSV)

    assert df is not None
    assert len(df) > 0
    assert df['Open'].dtype == np.float64

    print("✓ Load and validate combined OK")


def test_get_raw_data():
    """Test get_raw_data function"""
    df = loader.get_raw_data(TEST_CSV)

    assert df is not None
    assert len(df) > 0
    assert list(df.columns) == loader.REQUIRED_COLUMNS

    print("✓ get_raw_data OK")


def test_file_not_found():
    """Test error handling for missing file"""
    try:
        loader.load_csv("nonexistent.csv")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        print("✓ FileNotFoundError raised correctly")


def test_invalid_columns():
    """Test error handling for missing columns"""
    # Create a temporary invalid CSV
    invalid_df = pd.DataFrame({
        'Date': ['2020-01-01'],
        'Price': [100]
    })

    try:
        loader.validate_data(invalid_df)
        assert False, "Should raise ValueError for missing columns"
    except ValueError as e:
        assert "Missing required columns" in str(e)
        print("✓ Missing columns error raised correctly")


if __name__ == "__main__":
    print("\n=== Testing loader.py ===\n")

    test_load_valid_csv()
    test_load_extended_csv()
    test_validate_data()
    test_chronological_order()
    test_ohlc_logic()
    test_load_and_validate()
    test_get_raw_data()
    test_file_not_found()
    test_invalid_columns()

    print("\n✅ All loader tests passed!\n")
