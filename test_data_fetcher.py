"""Tests for data_fetcher module."""

import os
import tempfile
from io import BytesIO
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

import pandas as pd
import pytest
import requests

from data_fetcher import download_fao_fpi_data, validate_excel_structure


class TestDownloadFaoFpiData:
    """Test cases for download_fao_fpi_data function."""

    @patch('data_fetcher.requests.get')
    def test_successful_download(self, mock_get):
        """Test successful download returns BytesIO object."""
        mock_response = Mock()
        mock_response.content = b'fake excel data'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_fao_fpi_data(cache_dir=temp_dir)
            
            assert isinstance(result, BytesIO)
            assert result.getvalue() == b'fake excel data'
            mock_get.assert_called_once_with(
                'https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data.xls',
                timeout=30
            )

    @patch('data_fetcher.requests.get')
    def test_successful_download_creates_cache_file(self, mock_get):
        """Test that successful download saves file to cache."""
        mock_response = Mock()
        mock_response.content = b'fake excel data'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            download_fao_fpi_data(cache_dir=temp_dir)
            
            cache_files = list(Path(temp_dir).glob('fao_fpi_data_*.xls'))
            assert len(cache_files) == 1
            
            with open(cache_files[0], 'rb') as f:
                assert f.read() == b'fake excel data'

    @patch('data_fetcher.requests.get')
    @patch('data_fetcher.time.sleep')
    def test_timeout_retry_logic(self, mock_sleep, mock_get):
        """Test timeout scenario with retry logic."""
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout()
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(requests.exceptions.Timeout):
                download_fao_fpi_data(cache_dir=temp_dir)
            
            assert mock_get.call_count == 4  # Initial + 3 retries
            assert mock_sleep.call_count == 3  # Sleep before each retry
            mock_sleep.assert_any_call(1)
            mock_sleep.assert_any_call(2)
            mock_sleep.assert_any_call(4)

    @patch('data_fetcher.requests.get')
    @patch('data_fetcher.time.sleep')
    def test_retry_success_on_second_attempt(self, mock_sleep, mock_get):
        """Test successful download after one retry."""
        mock_response = Mock()
        mock_response.content = b'fake excel data'
        mock_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            mock_response
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_fao_fpi_data(cache_dir=temp_dir)
            
            assert isinstance(result, BytesIO)
            assert result.getvalue() == b'fake excel data'
            assert mock_get.call_count == 2
            mock_sleep.assert_called_once_with(1)

    @patch('data_fetcher.requests.get')
    def test_404_error_handling(self, mock_get):
        """Test 404 error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(requests.exceptions.HTTPError):
                download_fao_fpi_data(cache_dir=temp_dir)
            
            mock_get.assert_called_once()

    @patch('data_fetcher.requests.get')
    def test_connection_error_handling(self, mock_get):
        """Test connection error handling with retries."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(requests.exceptions.ConnectionError):
                download_fao_fpi_data(cache_dir=temp_dir)
            
            assert mock_get.call_count == 4  # Initial + 3 retries

    @patch('data_fetcher.requests.get')
    def test_cache_directory_creation(self, mock_get):
        """Test that cache directory is created if it doesn't exist."""
        mock_response = Mock()
        mock_response.content = b'fake excel data'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, 'new_cache_dir')
            assert not os.path.exists(cache_dir)
            
            download_fao_fpi_data(cache_dir=cache_dir)
            
            assert os.path.exists(cache_dir)
            cache_files = list(Path(cache_dir).glob('fao_fpi_data_*.xls'))
            assert len(cache_files) == 1

    @patch('data_fetcher.requests.get')
    def test_default_cache_directory(self, mock_get):
        """Test that default cache directory 'cache' is used when not specified."""
        mock_response = Mock()
        mock_response.content = b'fake excel data'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch('data_fetcher.os.makedirs') as mock_makedirs:
            with patch('builtins.open', mock_open()) as mock_file:
                download_fao_fpi_data()
                
                mock_makedirs.assert_called_once_with('cache', exist_ok=True)

    @patch('data_fetcher.requests.get')
    def test_custom_url(self, mock_get):
        """Test download with custom URL."""
        mock_response = Mock()
        mock_response.content = b'fake excel data'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        custom_url = "https://example.com/custom_data.xls"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            download_fao_fpi_data(url=custom_url, cache_dir=temp_dir)
            
            mock_get.assert_called_once_with(custom_url, timeout=30)

    def test_timestamped_filename_format(self):
        """Test that cache filename includes timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('data_fetcher.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.content = b'fake excel data'
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                with patch('data_fetcher.datetime') as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = "20231215_143022"
                    
                    download_fao_fpi_data(cache_dir=temp_dir)
                    
                    expected_file = Path(temp_dir) / "fao_fpi_data_20231215_143022.xls"
                    assert expected_file.exists()


class TestValidateExcelStructure:
    """Test cases for validate_excel_structure function."""

    def create_mock_excel_file(self, sheets_data):
        """Helper method to create a mock Excel file with specified sheets and columns."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                for sheet_name, columns in sheets_data.items():
                    # Create a DataFrame with the specified columns and some dummy data
                    df = pd.DataFrame({col: [1, 2, 3] for col in columns})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Read the file back into a BytesIO object
            with open(tmp_file.name, 'rb') as f:
                excel_data = BytesIO(f.read())
            
            # Clean up the temporary file
            os.unlink(tmp_file.name)
            return excel_data

    def test_valid_excel_structure(self):
        """Test validation of properly structured Excel file."""
        sheets_data = {
            'Annual': ['Date', 'Food Price Index', 'Meat Price Index', 'Dairy Products', 
                      'Cereals', 'Vegetable Oils', 'Sugar'],
            'Monthly': ['Date', 'Value']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == True
        assert error_message == ""

    def test_missing_annual_sheet(self):
        """Test validation when Annual sheet is missing."""
        sheets_data = {
            'Monthly': ['Date', 'Value'],
            'SomeOtherSheet': ['Col1', 'Col2']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == False
        assert "Missing required sheet(s): Annual" in error_message

    def test_missing_monthly_sheet(self):
        """Test validation when Monthly sheet is missing."""
        sheets_data = {
            'Annual': ['Date', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == False
        assert "Missing required sheet(s): Monthly" in error_message

    def test_missing_both_required_sheets(self):
        """Test validation when both required sheets are missing."""
        sheets_data = {
            'SomeSheet': ['Col1', 'Col2'],
            'AnotherSheet': ['ColA', 'ColB']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == False
        assert "Missing required sheet(s): Annual, Monthly" in error_message

    def test_annual_sheet_missing_required_columns(self):
        """Test validation when Annual sheet is missing required columns."""
        sheets_data = {
            'Annual': ['Date', 'Food Price Index', 'Meat'],  # Missing Dairy, Cereals, Oils, Sugar
            'Monthly': ['Date', 'Value']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == False
        assert "Annual sheet missing required column(s):" in error_message
        assert "Dairy" in error_message
        assert "Cereals" in error_message
        assert "Oils" in error_message
        assert "Sugar" in error_message

    def test_case_insensitive_column_matching(self):
        """Test that column matching is case-insensitive."""
        sheets_data = {
            'Annual': ['date', 'FOOD PRICE INDEX', 'meat price', 'dairy products', 
                      'cereals index', 'vegetable oils', 'sugar prices'],
            'Monthly': ['Date', 'Value']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == True
        assert error_message == ""

    def test_partial_column_name_matching(self):
        """Test that partial column name matching works (substring matching)."""
        sheets_data = {
            'Annual': ['Date', 'Overall Food Price Index Value', 'Meat Products Index', 
                      'Dairy and Milk Products', 'Cereals and Grains', 'Vegetable Oils Index', 
                      'Sugar and Sweeteners'],
            'Monthly': ['Date', 'Value']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == True
        assert error_message == ""

    def test_empty_excel_file(self):
        """Test validation of empty Excel file."""
        empty_data = BytesIO(b"")
        is_valid, error_message = validate_excel_structure(empty_data)
        
        assert is_valid == False
        assert "not a valid excel format" in error_message.lower() or "unexpected error" in error_message.lower()

    def test_invalid_excel_format(self):
        """Test validation of non-Excel file content."""
        invalid_data = BytesIO(b"This is not Excel data")
        is_valid, error_message = validate_excel_structure(invalid_data)
        
        assert is_valid == False
        assert "not a valid excel format" in error_message.lower() or "excel file validation error" in error_message.lower()

    def test_corrupted_excel_file(self):
        """Test validation of corrupted Excel file."""
        # Create corrupted Excel-like data
        corrupted_data = BytesIO(b"PK\x03\x04corrupted excel data that looks like zip but isn't")
        is_valid, error_message = validate_excel_structure(corrupted_data)
        
        assert is_valid == False
        assert any(phrase in error_message.lower() for phrase in 
                  ["not a valid excel format", "excel file validation error", "unexpected error"])

    @patch('data_fetcher.pd.ExcelFile')
    def test_pandas_exception_handling(self, mock_excel_file):
        """Test handling of pandas exceptions during Excel reading."""
        mock_excel_file.side_effect = ValueError("Excel file format cannot be determined")
        
        excel_data = BytesIO(b"fake excel data")
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == False
        assert "File is not a valid Excel format" in error_message

    @patch('data_fetcher.pd.read_excel')
    @patch('data_fetcher.pd.ExcelFile')
    def test_empty_data_error_handling(self, mock_excel_file, mock_read_excel):
        """Test handling of EmptyDataError from pandas."""
        mock_excel_file.return_value.sheet_names = ['Annual', 'Monthly']
        mock_read_excel.side_effect = pd.errors.EmptyDataError("No columns to parse from file")
        
        excel_data = BytesIO(b"fake excel data")
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == False
        assert "Excel file appears to be empty" in error_message

    def test_byteio_position_reset(self):
        """Test that BytesIO position is properly reset for multiple validations."""
        sheets_data = {
            'Annual': ['Date', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
            'Monthly': ['Date', 'Value']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        
        # First validation
        is_valid1, error1 = validate_excel_structure(excel_data)
        
        # Second validation on same BytesIO object
        is_valid2, error2 = validate_excel_structure(excel_data)
        
        assert is_valid1 == True
        assert is_valid2 == True
        assert error1 == ""
        assert error2 == ""

    def test_excel_with_extra_sheets(self):
        """Test that Excel files with additional sheets beyond required ones are still valid."""
        sheets_data = {
            'Annual': ['Date', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
            'Monthly': ['Date', 'Value'],
            'ExtraSheet1': ['Col1', 'Col2'],
            'ExtraSheet2': ['ColA', 'ColB']
        }
        
        excel_data = self.create_mock_excel_file(sheets_data)
        is_valid, error_message = validate_excel_structure(excel_data)
        
        assert is_valid == True
        assert error_message == ""