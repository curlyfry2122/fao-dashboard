"""Tests for excel_parser module."""

import os
import tempfile
from io import BytesIO
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from excel_parser import parse_fao_excel_data


class TestExcelParser:
    """Test cases for FAO Excel data parsing functionality."""

    def create_test_excel_file(self, sheet_data, header_row=0):
        """Helper method to create test Excel files with specified data structure."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                for sheet_name, data in sheet_data.items():
                    if header_row == 0:
                        # Simple case - headers at row 0
                        df = pd.DataFrame(data['data'], columns=data['columns'])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        # Complex case - headers at specified row with empty rows above
                        worksheet = writer.book.create_sheet(sheet_name)
                        
                        # Write empty rows
                        for row_idx in range(header_row):
                            for col_idx in range(len(data['columns'])):
                                worksheet.cell(row=row_idx + 1, column=col_idx + 1, value='')
                        
                        # Write headers at the specified row
                        for col_idx, col_name in enumerate(data['columns']):
                            worksheet.cell(row=header_row + 1, column=col_idx + 1, value=col_name)
                        
                        # Write data rows
                        for data_row_idx, data_row in enumerate(data['data']):
                            for col_idx, value in enumerate(data_row):
                                worksheet.cell(row=header_row + 2 + data_row_idx, column=col_idx + 1, value=value)
            
            # Read the file back into a BytesIO object
            with open(tmp_file.name, 'rb') as f:
                excel_data = BytesIO(f.read())
            
            # Clean up the temporary file
            os.unlink(tmp_file.name)
            return excel_data

    def test_parse_annual_data_basic(self):
        """Test basic parsing of Annual sheet with standard structure."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat Price Index', 'Dairy Products', 
                           'Cereals Index', 'Vegetable Oils', 'Sugar Index'],
                'data': [
                    [2020, 100.5, 95.2, 103.7, 99.8, 110.3, 85.6],
                    [2021, 120.8, 110.4, 125.9, 115.2, 140.7, 98.3],
                    [2022, 135.2, 125.7, 140.8, 130.5, 155.9, 110.7]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        expected_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        assert list(result_df.columns) == expected_columns
        assert len(result_df) == 3
        
        # Check first row values
        assert result_df.iloc[0]['date'] == pd.Timestamp('2020-01-01')
        assert result_df.iloc[0]['food_price_index'] == 100.5
        assert result_df.iloc[0]['meat'] == 95.2
        assert result_df.iloc[0]['dairy'] == 103.7
        assert result_df.iloc[0]['cereals'] == 99.8
        assert result_df.iloc[0]['oils'] == 110.3
        assert result_df.iloc[0]['sugar'] == 85.6

    def test_parse_monthly_data_basic(self):
        """Test basic parsing of Monthly sheet with date parsing."""
        sheet_data = {
            'Monthly': {
                'columns': ['Date', 'Food Price Index', 'Meat Index', 'Dairy Products Index', 
                           'Cereals', 'Oils', 'Sugar'],
                'data': [
                    ['2023-01', 142.5, 135.8, 148.2, 140.6, 165.3, 120.4],
                    ['2023-02', 145.1, 138.9, 151.7, 143.2, 168.9, 123.8],
                    ['2023-03', 148.6, 142.3, 155.4, 146.8, 172.5, 127.2]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Monthly')
        
        expected_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        assert list(result_df.columns) == expected_columns
        assert len(result_df) == 3
        
        # Check date parsing
        assert result_df.iloc[0]['date'] == pd.Timestamp('2023-01-01')
        assert result_df.iloc[1]['date'] == pd.Timestamp('2023-02-01')
        assert result_df.iloc[2]['date'] == pd.Timestamp('2023-03-01')

    def test_header_detection_row_1(self):
        """Test header detection when headers are in row 1 (0-indexed)."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [
                    [2020, 100.5, 95.2, 103.7, 99.8, 110.3, 85.6],
                    [2021, 120.8, 110.4, 125.9, 115.2, 140.7, 98.3]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data, header_row=1)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        assert len(result_df) == 2
        assert result_df.iloc[0]['food_price_index'] == 100.5

    def test_header_detection_row_2(self):
        """Test header detection when headers are in row 2 (0-indexed)."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [
                    [2020, 100.5, 95.2, 103.7, 99.8, 110.3, 85.6]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data, header_row=2)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        assert len(result_df) == 1
        assert result_df.iloc[0]['food_price_index'] == 100.5

    def test_missing_values_handling(self):
        """Test handling of missing/NaN values in data."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [
                    [2020, 100.5, None, 103.7, '', 110.3, 85.6],
                    [2021, '', 110.4, 125.9, 115.2, None, 98.3],
                    [2022, 135.2, 125.7, '', 130.5, 155.9, '']
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        # Check that missing values are converted to NaN
        assert pd.isna(result_df.iloc[0]['meat'])
        assert pd.isna(result_df.iloc[0]['cereals'])
        assert pd.isna(result_df.iloc[1]['food_price_index'])
        assert pd.isna(result_df.iloc[1]['oils'])
        assert pd.isna(result_df.iloc[2]['dairy'])
        assert pd.isna(result_df.iloc[2]['sugar'])
        
        # Check that valid values are preserved
        assert result_df.iloc[0]['food_price_index'] == 100.5
        assert result_df.iloc[1]['meat'] == 110.4
        assert result_df.iloc[2]['oils'] == 155.9

    def test_column_name_variations(self):
        """Test parsing with various column name formats."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Overall Food Price Index', 'Meat Products Index', 
                           'Dairy and Milk Products', 'Cereals and Grains', 'Vegetable Oils Index', 
                           'Sugar and Sweeteners Index'],
                'data': [
                    [2020, 100.5, 95.2, 103.7, 99.8, 110.3, 85.6]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        expected_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        assert list(result_df.columns) == expected_columns
        assert result_df.iloc[0]['food_price_index'] == 100.5

    def test_case_insensitive_column_matching(self):
        """Test that column matching is case-insensitive."""
        sheet_data = {
            'Annual': {
                'columns': ['year', 'FOOD PRICE INDEX', 'meat index', 'DAIRY products', 
                           'cereals', 'oils index', 'Sugar'],
                'data': [
                    [2020, 100.5, 95.2, 103.7, 99.8, 110.3, 85.6]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        assert result_df.iloc[0]['food_price_index'] == 100.5
        assert result_df.iloc[0]['meat'] == 95.2

    def test_date_parsing_various_formats(self):
        """Test parsing different date formats."""
        sheet_data = {
            'Monthly': {
                'columns': ['Date', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [
                    ['2023-01-01', 100.5, 95.2, 103.7, 99.8, 110.3, 85.6],
                    ['Jan 2023', 102.1, 96.8, 105.2, 101.3, 112.7, 87.9],
                    ['2023/02', 104.3, 98.5, 107.1, 103.6, 115.2, 89.4],
                    [2023, 106.8, 100.2, 109.5, 105.9, 117.8, 91.2]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Monthly')
        
        # Check that dates are parsed correctly
        assert result_df.iloc[0]['date'] == pd.Timestamp('2023-01-01')
        assert result_df.iloc[1]['date'] == pd.Timestamp('2023-01-01')  # Jan 2023 -> 2023-01-01
        assert result_df.iloc[2]['date'] == pd.Timestamp('2023-02-01')  # 2023/02 -> 2023-02-01
        assert result_df.iloc[3]['date'] == pd.Timestamp('2023-01-01')  # 2023 -> 2023-01-01

    def test_malformed_date_handling(self):
        """Test handling of unparseable dates."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [
                    ['invalid_date', 100.5, 95.2, 103.7, 99.8, 110.3, 85.6],
                    ['', 102.1, 96.8, 105.2, 101.3, 112.7, 87.9],
                    [2020, 104.3, 98.5, 107.1, 103.6, 115.2, 89.4]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        # Check that unparseable dates become NaT (Not a Time)
        assert pd.isna(result_df.iloc[0]['date'])
        assert pd.isna(result_df.iloc[1]['date'])
        assert result_df.iloc[2]['date'] == pd.Timestamp('2020-01-01')

    def test_non_numeric_value_conversion(self):
        """Test conversion of non-numeric values to NaN."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [
                    [2020, 'not_a_number', 95.2, 'invalid', 99.8, 110.3, 85.6],
                    [2021, 120.8, 'text', 125.9, 115.2, 'another_invalid', 98.3]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        # Check that non-numeric values are converted to NaN
        assert pd.isna(result_df.iloc[0]['food_price_index'])
        assert pd.isna(result_df.iloc[0]['dairy'])
        assert pd.isna(result_df.iloc[1]['meat'])
        assert pd.isna(result_df.iloc[1]['oils'])
        
        # Check that valid numeric values are preserved
        assert result_df.iloc[0]['meat'] == 95.2
        assert result_df.iloc[1]['food_price_index'] == 120.8

    def test_missing_columns_handling(self):
        """Test handling when some expected columns are missing."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat'],  # Missing dairy, cereals, oils, sugar
                'data': [
                    [2020, 100.5, 95.2],
                    [2021, 120.8, 110.4]
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        expected_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        assert list(result_df.columns) == expected_columns
        
        # Check that missing columns are filled with NaN
        assert pd.isna(result_df.iloc[0]['dairy'])
        assert pd.isna(result_df.iloc[0]['cereals'])
        assert pd.isna(result_df.iloc[0]['oils'])
        assert pd.isna(result_df.iloc[0]['sugar'])
        
        # Check that present columns have correct values
        assert result_df.iloc[0]['food_price_index'] == 100.5
        assert result_df.iloc[0]['meat'] == 95.2

    def test_empty_sheet_handling(self):
        """Test handling of empty Excel sheets."""
        sheet_data = {
            'Annual': {
                'columns': [],
                'data': []
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        expected_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        assert list(result_df.columns) == expected_columns
        assert len(result_df) == 0

    def test_no_data_rows_after_header(self):
        """Test handling when there are headers but no data rows."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': []  # No data rows
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        expected_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        assert list(result_df.columns) == expected_columns
        assert len(result_df) == 0

    def test_invalid_sheet_name(self):
        """Test handling of non-existent sheet names."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [[2020, 100.5, 95.2, 103.7, 99.8, 110.3, 85.6]]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        
        with pytest.raises(ValueError, match="Sheet 'NonExistent' not found"):
            parse_fao_excel_data(excel_data, 'NonExistent')

    def test_numeric_string_conversion(self):
        """Test that numeric strings are properly converted to float."""
        sheet_data = {
            'Annual': {
                'columns': ['Year', 'Food Price Index', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar'],
                'data': [
                    ['2020', '100.5', '95.2', '103.7', '99.8', '110.3', '85.6']
                ]
            }
        }
        
        excel_data = self.create_test_excel_file(sheet_data)
        result_df = parse_fao_excel_data(excel_data, 'Annual')
        
        assert result_df.iloc[0]['date'] == pd.Timestamp('2020-01-01')
        assert result_df.iloc[0]['food_price_index'] == 100.5
        assert result_df.iloc[0]['meat'] == 95.2
        
        # Verify all non-date columns are float dtype
        for col in ['food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']:
            assert result_df[col].dtype == 'float64'