#!/usr/bin/env python3
"""Test script for the pivot interface functionality."""

import pandas as pd
import sys
from datetime import datetime
from pivot_builder import prepare_temporal_dimensions, validate_pivot_size, create_pivot_table


def create_test_data():
    """Create sample FAO data for testing."""
    # Create 24 months of data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
    
    # Generate realistic-looking price index data
    base_prices = {
        'food_price_index': 120,
        'meat': 115,
        'dairy': 125,
        'cereals': 110,
        'oils': 140,
        'sugar': 105
    }
    
    data = {'date': dates}
    
    for index, base_price in base_prices.items():
        # Add some realistic variation over time
        trend = range(len(dates))
        noise = [i * 0.5 + (i % 3) * 2 - 1 for i in range(len(dates))]
        prices = [base_price + t * 0.3 + n for t, n in zip(trend, noise)]
        data[index] = prices
    
    return pd.DataFrame(data)


def test_temporal_dimensions():
    """Test temporal dimension preparation."""
    print("Testing temporal dimensions...")
    
    df = create_test_data()
    df_with_dims = prepare_temporal_dimensions(df)
    
    # Check if new columns were added
    expected_cols = ['Year', 'Quarter', 'Month']
    for col in expected_cols:
        assert col in df_with_dims.columns, f"Missing column: {col}"
    
    # Check some sample values
    assert df_with_dims['Year'].iloc[0] == 2022
    assert df_with_dims['Quarter'].iloc[0] == 'Q1'
    assert df_with_dims['Month'].iloc[0] == '2022-01'
    
    print("✅ Temporal dimensions test passed")


def test_pivot_validation():
    """Test pivot size validation."""
    print("Testing pivot validation...")
    
    df = create_test_data()
    df_with_dims = prepare_temporal_dimensions(df)
    
    # Test valid pivot (should be fine)
    is_valid, cells, msg = validate_pivot_size(df_with_dims, 'Year', ['food_price_index', 'meat'])
    assert is_valid, f"Valid pivot rejected: {msg}"
    assert cells <= 1000, f"Cell count incorrect: {cells}"
    
    # Test oversized pivot (create artificial large dataset)
    large_cols = [f'index_{i}' for i in range(600)]  # 600 columns
    is_valid, cells, msg = validate_pivot_size(df_with_dims, 'Month', large_cols)
    assert not is_valid, f"Oversized pivot accepted: {msg}"
    assert cells > 1000, f"Cell count should be > 1000: {cells}"
    
    print("✅ Pivot validation test passed")


def test_pivot_creation():
    """Test pivot table creation."""
    print("Testing pivot table creation...")
    
    df = create_test_data()
    
    index_mapping = {
        'Food Price Index': 'food_price_index',
        'Meat': 'meat',
        'Dairy': 'dairy',
        'Cereals': 'cereals',
        'Oils': 'oils',
        'Sugar': 'sugar'
    }
    
    # Test pivot by year
    pivot_df = create_pivot_table(
        df, 
        'Year', 
        ['Food Price Index', 'Meat'], 
        'mean',
        index_mapping
    )
    
    assert not pivot_df.empty, "Pivot table is empty"
    assert 'Year' in pivot_df.columns, "Year column missing"
    assert len(pivot_df) == 2, f"Expected 2 years, got {len(pivot_df)}"  # 2022, 2023
    
    # Test pivot by quarter
    pivot_df_q = create_pivot_table(
        df,
        'Quarter',
        ['Food Price Index'],
        'max',
        index_mapping
    )
    
    assert not pivot_df_q.empty, "Quarterly pivot table is empty"
    assert 'Quarter' in pivot_df_q.columns, "Quarter column missing"
    assert len(pivot_df_q) == 4, f"Expected 4 quarters, got {len(pivot_df_q)}"  # Q1-Q4
    
    print("✅ Pivot creation test passed")


def test_aggregation_functions():
    """Test different aggregation functions."""
    print("Testing aggregation functions...")
    
    df = create_test_data()
    
    index_mapping = {
        'Food Price Index': 'food_price_index',
        'Meat': 'meat'
    }
    
    # Test all aggregation functions
    for agg_func in ['mean', 'max', 'min']:
        pivot_df = create_pivot_table(
            df,
            'Year',
            ['Food Price Index'],
            agg_func,
            index_mapping
        )
        
        assert not pivot_df.empty, f"Pivot with {agg_func} aggregation is empty"
        assert 'Food Price Index' in pivot_df.columns, f"Price index column missing for {agg_func}"
        
        # Check that values are reasonable (not NaN, not zero)
        values = pivot_df['Food Price Index'].dropna()
        assert len(values) > 0, f"No valid values for {agg_func}"
        assert all(v > 0 for v in values), f"Invalid values for {agg_func}: {values.tolist()}"
    
    print("✅ Aggregation functions test passed")


def run_all_tests():
    """Run all tests."""
    print("🧪 Starting pivot interface tests...\n")
    
    try:
        test_temporal_dimensions()
        test_pivot_validation()
        test_pivot_creation()
        test_aggregation_functions()
        
        print("\n✅ All tests passed! Pivot interface is working correctly.")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)