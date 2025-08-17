"""Tests for excel_exporter module."""

import pytest
import pandas as pd
import xlsxwriter
from datetime import datetime, date
import tempfile
import os
from io import BytesIO

from excel_exporter import ExcelExporter


@pytest.fixture
def sample_fao_dataframe():
    """Sample FAO DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=12, freq='MS')
    return pd.DataFrame({
        'Date': dates,
        'Food Price Index': [100.5, 101.2, 102.8, 98.9, 103.5, 105.1, 
                           106.8, 107.2, 108.9, 110.1, 111.5, 112.3],
        'Meat': [95.2, 96.8, 97.1, 94.5, 98.2, 99.8, 
                101.2, 102.5, 103.1, 104.8, 105.2, 106.1],
        'Change %': [0.5, 0.7, 1.6, -3.8, 4.7, 1.5,
                    1.6, 0.4, 1.6, 1.1, 1.3, 0.7]
    })


@pytest.fixture
def exporter():
    """ExcelExporter instance for testing."""
    return ExcelExporter()


def test_excel_exporter_initialization(exporter):
    """Test ExcelExporter initialization."""
    assert isinstance(exporter, ExcelExporter)
    assert 'header' in exporter.default_formats
    assert 'number' in exporter.default_formats
    assert 'percentage' in exporter.default_formats
    assert exporter.default_formats['header']['bold'] is True


def test_generate_data_sheet_basic_functionality(exporter, sample_fao_dataframe):
    """Test basic sheet creation functionality."""
    # Test with valid data
    workbook = exporter.generate_data_sheet(sample_fao_dataframe, 'Test Sheet')
    
    # Verify workbook is created
    assert isinstance(workbook, xlsxwriter.Workbook)
    
    # Verify workbook can be closed without errors
    workbook.close()


def test_generate_data_sheet_with_empty_dataframe(exporter):
    """Test sheet creation with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    workbook = exporter.generate_data_sheet(empty_df, 'Empty Sheet')
    
    # Should create workbook without errors
    assert isinstance(workbook, xlsxwriter.Workbook)
    workbook.close()


def test_generate_data_sheet_input_validation(exporter):
    """Test input validation for generate_data_sheet method."""
    # Test with invalid DataFrame
    with pytest.raises(TypeError):
        exporter.generate_data_sheet("not a dataframe", "Test")
    
    # Test with empty sheet name
    df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(ValueError):
        exporter.generate_data_sheet(df, "")
    
    # Test with None sheet name
    with pytest.raises(ValueError):
        exporter.generate_data_sheet(df, None)


def test_sanitize_sheet_name(exporter):
    """Test sheet name sanitization."""
    # Test with invalid characters
    sanitized = exporter._sanitize_sheet_name("Test[Sheet]:*?/\\")
    assert sanitized == "Test_Sheet______"
    
    # Test with long name
    long_name = "A" * 40
    sanitized = exporter._sanitize_sheet_name(long_name)
    assert len(sanitized) <= 31
    
    # Test with valid name
    valid_name = "Valid Sheet Name"
    sanitized = exporter._sanitize_sheet_name(valid_name)
    assert sanitized == valid_name


def test_column_format_detection(exporter, sample_fao_dataframe):
    """Test automatic column format detection."""
    formats = {'percentage': 'pct', 'number': 'num', 'date': 'date', 'integer': 'int'}
    
    # Test percentage detection
    change_series = sample_fao_dataframe['Change %']
    format_result = exporter._get_column_format(change_series, 'Change %', formats)
    assert format_result == 'pct'
    
    # Test date detection
    date_series = sample_fao_dataframe['Date']
    format_result = exporter._get_column_format(date_series, 'Date', formats)
    assert format_result == 'date'
    
    # Test number detection
    number_series = sample_fao_dataframe['Food Price Index']
    format_result = exporter._get_column_format(number_series, 'Food Price Index', formats)
    assert format_result == 'num'


def test_workbook_return_type(exporter, sample_fao_dataframe):
    """Test that workbook object is returned correctly."""
    workbook = exporter.generate_data_sheet(sample_fao_dataframe, 'Return Test')
    
    # Verify it's an xlsxwriter Workbook
    assert isinstance(workbook, xlsxwriter.Workbook)
    
    # Verify workbook has expected methods
    assert hasattr(workbook, 'close')
    assert hasattr(workbook, 'add_worksheet')
    assert hasattr(workbook, 'add_format')
    
    workbook.close()


def test_formatting_applied_verification(exporter):
    """Test that formatting is properly applied to sheets."""
    # Create test data with different data types
    test_df = pd.DataFrame({
        'Text': ['A', 'B', 'C'],
        'Integer': [1, 2, 3],
        'Float': [1.1, 2.2, 3.3],
        'Percentage': [0.1, 0.2, 0.3],
        'Date': pd.date_range('2020-01-01', periods=3)
    })
    
    workbook = exporter.generate_data_sheet(test_df, 'Format Test')
    
    # Verify workbook creation successful
    assert isinstance(workbook, xlsxwriter.Workbook)
    
    # Test that default formats are properly created
    assert 'header' in exporter.default_formats
    assert exporter.default_formats['header']['bold'] is True
    assert exporter.default_formats['number']['num_format'] == '#,##0.0'
    assert exporter.default_formats['percentage']['num_format'] == '0.0%'
    
    workbook.close()


def test_create_formats(exporter):
    """Test format creation from default formats."""
    # Create a test workbook to test format creation
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    
    formats = exporter._create_formats(workbook)
    
    # Verify all expected formats are created
    assert 'header' in formats
    assert 'number' in formats
    assert 'percentage' in formats
    assert 'date' in formats
    assert 'integer' in formats
    
    # Verify formats are format objects (can't check exact type due to xlsxwriter internals)
    for format_obj in formats.values():
        assert format_obj is not None
        assert hasattr(format_obj, 'num_format')
    
    workbook.close()


def test_column_width_adjustment(exporter):
    """Test automatic column width adjustment."""
    # Create DataFrame with varying content lengths
    test_df = pd.DataFrame({
        'Short': ['A', 'B'],
        'Medium Length Column': ['Medium', 'Content'],
        'Very Long Column Name With Lots Of Text': ['Long content here', 'More long content']
    })
    
    workbook = exporter.generate_data_sheet(test_df, 'Width Test')
    
    # Should complete without errors
    assert isinstance(workbook, xlsxwriter.Workbook)
    workbook.close()


def test_freeze_panes_and_autofilter_options(exporter, sample_fao_dataframe):
    """Test freeze panes and autofilter options."""
    # Test with freeze_panes=True (default)
    workbook1 = exporter.generate_data_sheet(
        sample_fao_dataframe, 
        'Frozen Test', 
        freeze_panes=True, 
        auto_filter=True
    )
    assert isinstance(workbook1, xlsxwriter.Workbook)
    workbook1.close()
    
    # Test with freeze_panes=False
    workbook2 = exporter.generate_data_sheet(
        sample_fao_dataframe, 
        'No Freeze Test', 
        freeze_panes=False, 
        auto_filter=False
    )
    assert isinstance(workbook2, xlsxwriter.Workbook)
    workbook2.close()


def test_edge_cases_and_data_types(exporter):
    """Test edge cases and various data types."""
    # DataFrame with NaN values, mixed types
    edge_case_df = pd.DataFrame({
        'Mixed': [1, 'text', None, 3.14],
        'NaN Values': [1.0, float('nan'), 3.0, float('nan')],
        'Boolean': [True, False, True, False],
        'Large Numbers': [1000000.123, 2000000.456, 3000000.789, 4000000.012]
    })
    
    workbook = exporter.generate_data_sheet(edge_case_df, 'Edge Cases')
    
    # Should handle edge cases without errors
    assert isinstance(workbook, xlsxwriter.Workbook)
    workbook.close()


def test_integration_with_fao_data_structure(exporter):
    """Test integration with FAO-specific data structures."""
    # Simulate real FAO data structure
    fao_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=6, freq='MS'),
        'food_price_index': [118.9, 119.5, 120.1, 118.8, 121.4, 122.1],
        'meat': [112.5, 113.1, 114.2, 113.8, 115.5, 116.2],
        'dairy': [125.8, 126.4, 127.1, 125.9, 128.2, 129.1],
        'cereals': [108.2, 109.1, 110.3, 109.8, 111.2, 112.5],
        'oils': [134.5, 135.2, 136.8, 135.1, 137.9, 138.4],
        'sugar': [98.7, 99.2, 100.1, 99.8, 101.3, 102.1],
        'food_price_index_yoy_change': [5.2, 4.8, 6.1, 5.9, 7.2, 6.8],
        'food_price_index_mom_change': [0.5, 0.3, 0.5, -1.1, 2.2, 0.6]
    })
    
    workbook = exporter.generate_data_sheet(fao_df, 'FAO Monthly Data')
    
    # Should handle FAO data structure correctly
    assert isinstance(workbook, xlsxwriter.Workbook)
    workbook.close()


def test_empty_dataframe_with_columns(exporter):
    """Test empty DataFrame that has columns but no data."""
    empty_with_columns = pd.DataFrame(columns=['Date', 'Price Index', 'Change %'])
    
    workbook = exporter.generate_data_sheet(empty_with_columns, 'Empty With Columns')
    
    # Should handle empty DataFrame with columns
    assert isinstance(workbook, xlsxwriter.Workbook)
    workbook.close()


if __name__ == "__main__":
    pytest.main([__file__])