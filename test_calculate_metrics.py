"""Tests for calculate_metrics module."""

import numpy as np
import pandas as pd
import pytest

from calculate_metrics import calculate_metrics


class TestCalculateMetrics:
    """Test cases for FAO metrics calculation functionality."""

    def create_sample_monthly_data(self, months=24, start_date='2022-01-01'):
        """Create sample monthly data for testing."""
        dates = pd.date_range(start=start_date, periods=months, freq='MS')
        
        # Create realistic FAO price index data with trends and seasonal patterns
        base_values = {
            'food_price_index': 100.0,
            'meat': 95.0,
            'dairy': 105.0,
            'cereals': 98.0,
            'oils': 110.0,
            'sugar': 85.0
        }
        
        data = {'date': dates}
        
        for col, base_val in base_values.items():
            # Create data with trend and some volatility
            trend = np.linspace(0, months * 2, months)  # 2% growth per month
            noise = np.random.normal(0, 5, months)  # Random volatility
            seasonal = 3 * np.sin(2 * np.pi * np.arange(months) / 12)  # Seasonal pattern
            
            values = base_val + trend + noise + seasonal
            data[col] = values
        
        return pd.DataFrame(data)

    def create_sample_annual_data(self, years=3, start_year=2020):
        """Create sample annual data for testing."""
        dates = [pd.Timestamp(f'{start_year + i}-01-01') for i in range(years)]
        
        # Create realistic annual data
        base_values = {
            'food_price_index': 100.0,
            'meat': 95.0,
            'dairy': 105.0,
            'cereals': 98.0,
            'oils': 110.0,
            'sugar': 85.0
        }
        
        data = {'date': dates}
        
        for col, base_val in base_values.items():
            # Annual growth pattern
            values = [base_val * (1.05 ** i) for i in range(years)]  # 5% annual growth
            data[col] = values
        
        return pd.DataFrame(data)

    def test_yoy_change_calculation_monthly(self):
        """Test year-over-year change calculation with monthly data."""
        df = self.create_sample_monthly_data(25)  # 25 months to have >12 months data
        
        # Set specific values for easier testing
        df.loc[0, 'food_price_index'] = 100.0  # Jan 2022
        df.loc[12, 'food_price_index'] = 110.0  # Jan 2023
        
        result_df = calculate_metrics(df, ['yoy_change'])
        
        # Check that YoY column exists
        assert 'food_price_index_yoy_change' in result_df.columns
        
        # First 12 months should have NaN for YoY change
        for i in range(12):
            assert pd.isna(result_df.iloc[i]['food_price_index_yoy_change'])
        
        # 13th month (index 12) should have YoY change: (110-100)/100*100 = 10%
        expected_yoy = (110.0 - 100.0) / 100.0 * 100
        actual_yoy = result_df.iloc[12]['food_price_index_yoy_change']
        assert abs(actual_yoy - expected_yoy) < 0.01

    def test_mom_change_calculation(self):
        """Test month-over-month change calculation."""
        df = self.create_sample_monthly_data(5)
        
        # Set specific values for easier testing
        df.loc[0, 'food_price_index'] = 100.0
        df.loc[1, 'food_price_index'] = 105.0
        df.loc[2, 'food_price_index'] = 102.0
        
        result_df = calculate_metrics(df, ['mom_change'])
        
        # Check that MoM column exists
        assert 'food_price_index_mom_change' in result_df.columns
        
        # First month should have NaN for MoM change
        assert pd.isna(result_df.iloc[0]['food_price_index_mom_change'])
        
        # Second month: (105-100)/100*100 = 5%
        expected_mom_1 = (105.0 - 100.0) / 100.0 * 100
        actual_mom_1 = result_df.iloc[1]['food_price_index_mom_change']
        assert abs(actual_mom_1 - expected_mom_1) < 0.01
        
        # Third month: (102-105)/105*100 = -2.857%
        expected_mom_2 = (102.0 - 105.0) / 105.0 * 100
        actual_mom_2 = result_df.iloc[2]['food_price_index_mom_change']
        assert abs(actual_mom_2 - expected_mom_2) < 0.01

    def test_12m_avg_calculation(self):
        """Test 12-month rolling average calculation."""
        df = self.create_sample_monthly_data(15)
        
        # Set specific values for first 12 months
        for i in range(12):
            df.loc[i, 'food_price_index'] = 100.0 + i  # 100, 101, 102, ..., 111
        
        result_df = calculate_metrics(df, ['12m_avg'])
        
        # Check that 12m_avg column exists
        assert 'food_price_index_12m_avg' in result_df.columns
        
        # First 11 months should have NaN for 12m average
        for i in range(11):
            assert pd.isna(result_df.iloc[i]['food_price_index_12m_avg'])
        
        # 12th month (index 11) should have average of first 12 values
        expected_avg = np.mean([100.0 + i for i in range(12)])  # (100+101+...+111)/12 = 105.5
        actual_avg = result_df.iloc[11]['food_price_index_12m_avg']
        assert abs(actual_avg - expected_avg) < 0.01

    def test_all_metrics_together(self):
        """Test calculating all metrics together."""
        df = self.create_sample_monthly_data(24)
        
        result_df = calculate_metrics(df, ['yoy_change', 'mom_change', '12m_avg'])
        
        # Check all columns exist (order doesn't matter for this test)
        expected_columns = [
            'date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar',
            'food_price_index_yoy_change', 'food_price_index_mom_change', 'food_price_index_12m_avg',
            'meat_yoy_change', 'meat_mom_change', 'meat_12m_avg',
            'dairy_yoy_change', 'dairy_mom_change', 'dairy_12m_avg',
            'cereals_yoy_change', 'cereals_mom_change', 'cereals_12m_avg',
            'oils_yoy_change', 'oils_mom_change', 'oils_12m_avg',
            'sugar_yoy_change', 'sugar_mom_change', 'sugar_12m_avg'
        ]
        
        for col in expected_columns:
            assert col in result_df.columns
        
        # Original data should be preserved
        pd.testing.assert_frame_equal(
            result_df[['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']],
            df[['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']]
        )

    def test_annual_data_yoy_change(self):
        """Test YoY change calculation with annual data."""
        df = self.create_sample_annual_data(3)
        
        # Set specific values
        df.loc[0, 'food_price_index'] = 100.0  # 2020
        df.loc[1, 'food_price_index'] = 110.0  # 2021
        df.loc[2, 'food_price_index'] = 121.0  # 2022
        
        result_df = calculate_metrics(df, ['yoy_change'])
        
        # First year should have NaN
        assert pd.isna(result_df.iloc[0]['food_price_index_yoy_change'])
        
        # Second year: (110-100)/100*100 = 10%
        expected_yoy_1 = (110.0 - 100.0) / 100.0 * 100
        actual_yoy_1 = result_df.iloc[1]['food_price_index_yoy_change']
        assert abs(actual_yoy_1 - expected_yoy_1) < 0.01
        
        # Third year: (121-110)/110*100 = 10%
        expected_yoy_2 = (121.0 - 110.0) / 110.0 * 100
        actual_yoy_2 = result_df.iloc[2]['food_price_index_yoy_change']
        assert abs(actual_yoy_2 - expected_yoy_2) < 0.01

    def test_missing_data_handling(self):
        """Test handling of missing data in calculations."""
        # Create deterministic data for consistent testing
        dates = pd.date_range('2022-01-01', periods=15, freq='MS')
        values = [100.0 + i for i in range(15)]  # 100, 101, 102, etc.
        
        df = pd.DataFrame({
            'date': dates,
            'food_price_index': values,
            'meat': values.copy(),
            'dairy': values.copy(),
            'cereals': values.copy(),
            'oils': values.copy(),
            'sugar': values.copy()
        })
        
        # Introduce some NaN values at specific positions
        df.loc[5, 'food_price_index'] = np.nan  # Index 5 = June 2022
        df.loc[10, 'meat'] = np.nan  # Index 10 = November 2022
        
        result_df = calculate_metrics(df, ['yoy_change', 'mom_change', '12m_avg'])
        
        # Should handle NaN gracefully without crashing
        assert len(result_df) == len(df)
        
        # NaN values should propagate appropriately
        # Index 5 (June) has NaN, so index 6 (July) MoM change should be NaN
        # because it can't calculate (July - NaN) / NaN
        assert pd.isna(result_df.iloc[5]['food_price_index_mom_change'])  # June change is NaN
        assert pd.isna(result_df.iloc[6]['food_price_index_mom_change'])  # July change is NaN

    def test_insufficient_data_for_yoy(self):
        """Test handling when insufficient data for YoY calculation."""
        df = self.create_sample_monthly_data(6)  # Only 6 months
        
        result_df = calculate_metrics(df, ['yoy_change'])
        
        # All YoY values should be NaN
        for i in range(len(result_df)):
            assert pd.isna(result_df.iloc[i]['food_price_index_yoy_change'])

    def test_insufficient_data_for_12m_avg(self):
        """Test handling when insufficient data for 12-month average."""
        df = self.create_sample_monthly_data(8)  # Only 8 months
        
        result_df = calculate_metrics(df, ['12m_avg'])
        
        # All 12m_avg values should be NaN
        for i in range(len(result_df)):
            assert pd.isna(result_df.iloc[i]['food_price_index_12m_avg'])

    def test_single_data_point(self):
        """Test handling with only one data point."""
        df = self.create_sample_monthly_data(1)
        
        result_df = calculate_metrics(df, ['yoy_change', 'mom_change', '12m_avg'])
        
        # All calculated values should be NaN
        assert pd.isna(result_df.iloc[0]['food_price_index_yoy_change'])
        assert pd.isna(result_df.iloc[0]['food_price_index_mom_change'])
        assert pd.isna(result_df.iloc[0]['food_price_index_12m_avg'])

    def test_unsorted_data_handling(self):
        """Test that function handles unsorted data correctly."""
        df = self.create_sample_monthly_data(12)
        
        # Shuffle the DataFrame
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        # Calculate metrics on shuffled data
        result_df = calculate_metrics(df_shuffled, ['mom_change'])
        
        # Should automatically sort by date and calculate correctly
        # After sorting, the result should be the same as with sorted data
        sorted_df = df.sort_values('date').reset_index(drop=True)
        expected_df = calculate_metrics(sorted_df, ['mom_change'])
        
        # Compare the sorted results
        result_sorted = result_df.sort_values('date').reset_index(drop=True)
        pd.testing.assert_frame_equal(result_sorted, expected_df)

    def test_invalid_metrics_error(self):
        """Test error handling for invalid metric names."""
        df = self.create_sample_monthly_data(12)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_metrics(df, ['invalid_metric'])

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar'])
        
        result_df = calculate_metrics(df, ['yoy_change', 'mom_change', '12m_avg'])
        
        # Should return empty DataFrame with all expected columns
        # Columns are added by metric type (all yoy_change, then mom_change, then 12m_avg)
        expected_columns = [
            'date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar',
            'food_price_index_yoy_change', 'meat_yoy_change', 'dairy_yoy_change', 
            'cereals_yoy_change', 'oils_yoy_change', 'sugar_yoy_change',
            'food_price_index_mom_change', 'meat_mom_change', 'dairy_mom_change',
            'cereals_mom_change', 'oils_mom_change', 'sugar_mom_change',
            'food_price_index_12m_avg', 'meat_12m_avg', 'dairy_12m_avg',
            'cereals_12m_avg', 'oils_12m_avg', 'sugar_12m_avg'
        ]
        
        assert list(result_df.columns) == expected_columns
        assert len(result_df) == 0

    def test_edge_case_exactly_12_months(self):
        """Test edge case with exactly 12 months of data."""
        df = self.create_sample_monthly_data(12)
        
        result_df = calculate_metrics(df, ['yoy_change', '12m_avg'])
        
        # YoY should be all NaN (need >12 months)
        for i in range(12):
            assert pd.isna(result_df.iloc[i]['food_price_index_yoy_change'])
        
        # 12m_avg should have value only for the last month
        for i in range(11):
            assert pd.isna(result_df.iloc[i]['food_price_index_12m_avg'])
        assert not pd.isna(result_df.iloc[11]['food_price_index_12m_avg'])

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy of calculations with known values."""
        # Create DataFrame with known values for precise testing
        dates = pd.date_range('2022-01-01', periods=14, freq='MS')
        values = [100, 102, 98, 105, 110, 108, 115, 120, 125, 118, 122, 130, 135, 140]
        
        df = pd.DataFrame({
            'date': dates,
            'food_price_index': values,
            'meat': values,  # Same values for simplicity
            'dairy': values,
            'cereals': values,
            'oils': values,
            'sugar': values
        })
        
        result_df = calculate_metrics(df, ['yoy_change', 'mom_change', '12m_avg'])
        
        # Test specific calculations
        # MoM change for Feb 2022: (102-100)/100*100 = 2%
        assert abs(result_df.iloc[1]['food_price_index_mom_change'] - 2.0) < 0.01
        
        # MoM change for Mar 2022: (98-102)/102*100 = -3.922%
        expected_mom_mar = (98 - 102) / 102 * 100
        assert abs(result_df.iloc[2]['food_price_index_mom_change'] - expected_mom_mar) < 0.01
        
        # YoY change for Jan 2023 (index 12): (135-100)/100*100 = 35%
        assert abs(result_df.iloc[12]['food_price_index_yoy_change'] - 35.0) < 0.01
        
        # 12m average for Dec 2022 (index 11): average of first 12 values
        expected_12m_avg = np.mean(values[:12])
        assert abs(result_df.iloc[11]['food_price_index_12m_avg'] - expected_12m_avg) < 0.01

    def test_data_types_preservation(self):
        """Test that appropriate data types are preserved/set."""
        df = self.create_sample_monthly_data(15)
        
        result_df = calculate_metrics(df, ['yoy_change', 'mom_change', '12m_avg'])
        
        # Date column should remain datetime
        assert pd.api.types.is_datetime64_any_dtype(result_df['date'])
        
        # All numeric columns should be float64
        numeric_columns = [col for col in result_df.columns if col != 'date']
        for col in numeric_columns:
            assert pd.api.types.is_float_dtype(result_df[col])

    def test_no_metrics_requested(self):
        """Test behavior when no metrics are requested."""
        df = self.create_sample_monthly_data(12)
        
        result_df = calculate_metrics(df, [])
        
        # Should return original DataFrame unchanged
        pd.testing.assert_frame_equal(result_df, df)