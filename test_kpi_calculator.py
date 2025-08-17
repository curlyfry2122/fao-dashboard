"""Tests for kpi_calculator module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from pathlib import Path
import tempfile
import pandas as pd

from kpi_calculator import calculate_kpis, get_kpi_summary, format_kpi_for_display, get_trend_emoji, get_delta_color, calculate_detailed_statistics

@pytest.fixture
def sample_fao_dataframe():
    """Sample FAO DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    return pd.DataFrame({
        'date': dates,
        'food_price_index': [100 + i*0.5 for i in range(24)],
        'meat': [95 + i*0.3 for i in range(24)],
        'dairy': [105 + i*0.7 for i in range(24)]
    })

def test_calculate_kpis(sample_fao_dataframe):
    """Test calculate_kpis function."""
    # Test with valid data
    result = calculate_kpis(sample_fao_dataframe, ['food_price_index', 'meat'], 'date')
    
    assert isinstance(result, dict)
    assert 'food_price_index' in result
    assert 'meat' in result
    
    # Check structure of each KPI
    for index in ['food_price_index', 'meat']:
        kpi = result[index]
        assert 'current_value' in kpi
        assert '12m_avg' in kpi
        assert 'yoy_change' in kpi
        assert 'trend_direction' in kpi
        
        # Verify current value is the latest value
        assert kpi['current_value'] == sample_fao_dataframe[index].iloc[-1]
    
    # Edge cases
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        calculate_kpis(pd.DataFrame(), ['food_price_index'], 'date')
    
    # Test with missing columns
    with pytest.raises(KeyError):
        calculate_kpis(sample_fao_dataframe, ['nonexistent_column'], 'date')

def test_get_kpi_summary():
    """Test get_kpi_summary function."""
    # Test with valid KPI data
    sample_kpis = {
        'food_price_index': {
            'current_value': 100.0,
            '12m_avg': 98.5,
            'yoy_change': 5.2,
            'trend_direction': 'up'
        },
        'meat': {
            'current_value': 95.0,
            '12m_avg': 94.0,
            'yoy_change': -2.1,
            'trend_direction': 'down'
        }
    }
    
    result = get_kpi_summary(sample_kpis)
    
    assert isinstance(result, dict)
    assert result['total_indices'] == 2
    assert result['indices_trending_up'] == 1
    assert result['indices_trending_down'] == 1
    assert result['avg_yoy_change'] is not None
    
    # Test with empty KPIs
    result_empty = get_kpi_summary({})
    assert result_empty['total_indices'] == 0
    assert result_empty['avg_yoy_change'] is None

def test_format_kpi_for_display():
    """Test format_kpi_for_display function."""
    # Test value formatting
    assert format_kpi_for_display(123.456, 'value', 1) == "123.5"
    assert format_kpi_for_display(123.456, 'percentage', 2) == "123.46%"
    assert format_kpi_for_display(None, 'value') == "N/A"
    
    # Test different decimal places
    assert format_kpi_for_display(100.0, 'value', 0) == "100"

def test_get_trend_emoji():
    """Test get_trend_emoji function."""
    # Test all trend directions
    assert get_trend_emoji('up') == '📈'
    assert get_trend_emoji('down') == '📉'
    assert get_trend_emoji('stable') == '➡️'
    assert get_trend_emoji('unknown') == '❓'
    
    # Test invalid/unknown trend
    assert get_trend_emoji('invalid') == '❓'

def test_get_delta_color():
    """Test get_delta_color function."""
    # Test with positive change (should be inverse for food prices)
    result_positive = get_delta_color(5.0)
    assert result_positive == 'inverse'
    
    # Test with negative change
    result_negative = get_delta_color(-3.0)
    assert result_negative == 'inverse'
    
    # Test with None
    result_none = get_delta_color(None)
    assert result_none is None

def test_calculate_detailed_statistics(sample_fao_dataframe):
    """Test calculate_detailed_statistics function."""
    # Sample index mapping
    index_mapping = {
        'Food Price Index': 'food_price_index',
        'Meat': 'meat',
        'Dairy': 'dairy'
    }
    
    # Test with valid data
    selected_indices = ['Food Price Index', 'Meat']
    result = calculate_detailed_statistics(sample_fao_dataframe, selected_indices, index_mapping)
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two selected indices
    assert list(result.columns) == ['Index', 'Min', 'Max', 'Mean', 'Std Dev', 'Sparkline']
    
    # Check data types and values
    for _, row in result.iterrows():
        assert isinstance(row['Index'], str)
        assert isinstance(row['Min'], float)
        assert isinstance(row['Max'], float) 
        assert isinstance(row['Mean'], float)
        assert isinstance(row['Std Dev'], float)
        assert isinstance(row['Sparkline'], list)
        assert len(row['Sparkline']) <= 24  # Default sparkline periods
        
        # Check logical constraints
        assert row['Min'] <= row['Mean'] <= row['Max']
        assert row['Std Dev'] >= 0
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        calculate_detailed_statistics(pd.DataFrame(), selected_indices, index_mapping)
    
    # Test with no valid indices
    with pytest.raises(ValueError):
        calculate_detailed_statistics(sample_fao_dataframe, [], index_mapping)