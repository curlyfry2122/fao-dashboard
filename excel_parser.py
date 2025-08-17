"""Module for parsing FAO Food Price Index Excel data."""

import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_fao_excel_data(excel_data: BytesIO, sheet_name: str) -> pd.DataFrame:
    """
    Parse FAO Excel data and return standardized DataFrame.
    
    Extracts data from the specified sheet and returns a standardized DataFrame
    with consistent column names and data types. Handles various Excel formats
    with intelligent header detection and robust data conversion.
    
    Args:
        excel_data: BytesIO object containing Excel file content.
        sheet_name: Name of the sheet to parse ('Annual' or 'Monthly').
        
    Returns:
        pd.DataFrame: Standardized DataFrame with columns:
            ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
            
    Raises:
        ValueError: If sheet_name is not found in the Excel file.
        
    Example:
        >>> from data_fetcher import download_fao_fpi_data
        >>> excel_data = download_fao_fpi_data()
        >>> df = parse_fao_excel_data(excel_data, 'Annual')
        >>> print(df.columns.tolist())
        ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
    """
    # Reset BytesIO position to beginning
    excel_data.seek(0)
    
    # Auto-detect Excel engine based on file format
    excel_file = None
    engines_to_try = ['xlrd', 'openpyxl']  # Try xlrd first for .xls, then openpyxl for .xlsx
    
    last_exception = None
    for engine in engines_to_try:
        try:
            excel_data.seek(0)  # Reset position for each attempt
            excel_file = pd.ExcelFile(excel_data, engine=engine)
            break  # Success, break out of loop
        except Exception as e:
            last_exception = e
            continue  # Try next engine
    
    if excel_file is None:
        raise ValueError(f"Unable to read Excel file with any supported engine. Last error: {str(last_exception)}")
    
    # Handle sheet name mapping for new FAO format
    sheet_name_mapping = {
        'Monthly': 'Indices_Monthly',  # New format uses Indices_Monthly
        'Annual': 'Annual'  # Annual stays the same
    }
    
    # Determine actual sheet name to use
    actual_sheet_name = sheet_name
    available_sheets = excel_file.sheet_names
    
    if sheet_name not in available_sheets:
        # Try mapped name for new format
        if sheet_name in sheet_name_mapping and sheet_name_mapping[sheet_name] in available_sheets:
            actual_sheet_name = sheet_name_mapping[sheet_name]
            print(f"📋 Using new format sheet: {actual_sheet_name} (requested: {sheet_name})")
        else:
            raise ValueError(f"Sheet '{sheet_name}' not found in Excel file. Available sheets: {available_sheets}")
    
    print(f"📊 Processing sheet: {actual_sheet_name}")
    
    # Determine the engine that worked
    engine_to_use = excel_file.engine.name if hasattr(excel_file.engine, 'name') else engines_to_try[0]
    
    # Read the sheet with multiple header row attempts
    df = None
    header_row = _detect_header_row(excel_data, actual_sheet_name, engine_to_use)
    
    if header_row is not None:
        excel_data.seek(0)  # Reset position
        df = pd.read_excel(excel_data, sheet_name=actual_sheet_name, header=header_row, engine=engine_to_use)
    else:
        # Fallback: try reading without header and create our own
        excel_data.seek(0)
        df = pd.read_excel(excel_data, sheet_name=actual_sheet_name, header=None, engine=engine_to_use)
        if len(df) == 0:
            df = pd.DataFrame()  # Empty DataFrame
    
    # If DataFrame is empty, return standardized empty DataFrame
    if df.empty:
        return _create_empty_standardized_dataframe()
    
    # Map columns to standardized names
    column_mapping = _map_columns(df.columns.tolist())
    
    # Create standardized DataFrame with the correct number of rows
    num_rows = len(df)
    standardized_data = {}
    
    # Define standardized columns
    std_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
    
    # Extract and convert data for each standardized column
    for std_col in std_columns:
        original_col = column_mapping.get(std_col)
        
        if original_col is not None and original_col in df.columns:
            if std_col == 'date':
                standardized_data[std_col] = _parse_dates(df[original_col])
            else:
                standardized_data[std_col] = _convert_to_float(df[original_col])
        else:
            # Create column with NaN values for missing columns
            if std_col == 'date':
                standardized_data[std_col] = pd.Series([pd.NaT] * num_rows)
            else:
                standardized_data[std_col] = pd.Series([np.nan] * num_rows, dtype='float64')
    
    standardized_df = pd.DataFrame(standardized_data)
    return standardized_df


def _detect_header_row(excel_data: BytesIO, sheet_name: str, engine: str = 'xlrd') -> Optional[int]:
    """
    Intelligently detect which row contains the headers.
    
    Tries rows 0, 1, and 2 and selects the one with the most expected columns.
    
    Args:
        excel_data: BytesIO object containing Excel file content.
        sheet_name: Name of the sheet to analyze.
        engine: Pandas Excel engine to use for reading.
        
    Returns:
        int or None: Row index containing headers, or None if not found.
    """
    expected_keywords = ['food price index', 'meat', 'dairy', 'cereals', 'oils', 'sugar', 'year', 'date']
    
    best_row = None
    best_score = 0
    
    for row_idx in range(3):  # Check rows 0, 1, 2
        try:
            excel_data.seek(0)
            # Read a few rows to get headers and check if they make sense
            df = pd.read_excel(excel_data, sheet_name=sheet_name, header=row_idx, nrows=3, engine=engine)
            
            if df.empty or len(df.columns) == 0:
                continue
                
            # Convert column names to lowercase for matching
            column_names_lower = [str(col).lower() for col in df.columns]
            
            # Count how many expected keywords are found in column names
            score = 0
            for keyword in expected_keywords:
                if any(keyword in col_name for col_name in column_names_lower):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_row = row_idx
                
        except Exception:
            # Skip this row if there's an error reading it
            continue
    
    # If no good header found but we detected some content, default to row 0
    return best_row if best_score > 0 else 0


def _map_columns(column_names: List[str]) -> Dict[str, Optional[str]]:
    """
    Map Excel column names to standardized column names.
    
    Uses fuzzy matching to handle various FAO column name formats.
    
    Args:
        column_names: List of column names from the Excel file.
        
    Returns:
        Dict mapping standardized names to original column names.
    """
    # Define mapping patterns (case-insensitive)
    mapping_patterns = {
        'date': [r'\byear\b', r'\bdate\b', r'\btime\b', r'\bperiod\b'],
        'food_price_index': [r'food.*price.*index', r'overall.*food', r'food.*index', r'\bfpi\b'],
        'meat': [r'\bmeat\b', r'\blivestock\b'],
        'dairy': [r'\bdairy\b', r'\bmilk\b'],
        'cereals': [r'\bcereals?\b', r'\bgrains?\b', r'\bwheat\b', r'\brice\b'],
        'oils': [r'\boils?\b', r'vegetable.*oils?', r'\bfats?\b'],
        'sugar': [r'\bsugar\b', r'\bsweeteners?\b']
    }
    
    column_mapping = {std_col: None for std_col in mapping_patterns.keys()}
    column_names_lower = [str(col).lower() for col in column_names]
    
    for std_col, patterns in mapping_patterns.items():
        best_match = None
        best_score = 0
        
        for i, col_name in enumerate(column_names_lower):
            score = 0
            for pattern in patterns:
                if re.search(pattern, col_name):
                    # Give higher score for more specific matches
                    score += len(pattern)
            
            if score > best_score:
                best_score = score
                best_match = column_names[i]  # Use original case
        
        column_mapping[std_col] = best_match
    
    return column_mapping


def _parse_dates(date_series: pd.Series) -> pd.Series:
    """
    Parse various date formats to standardized datetime.
    
    Handles formats like:
    - 2020 (year only) -> 2020-01-01
    - 2023-01 -> 2023-01-01
    - Jan 2023 -> 2023-01-01
    - 2023-01-15 -> 2023-01-15
    
    Args:
        date_series: Pandas Series containing date values.
        
    Returns:
        pd.Series: Series with parsed datetime values.
    """
    parsed_dates = []
    
    for date_val in date_series:
        parsed_date = None
        
        if pd.isna(date_val):
            parsed_date = pd.NaT
        else:
            # Convert to string for processing
            date_str = str(date_val).strip()
            
            if not date_str or date_str.lower() in ['', 'nan', 'none']:
                parsed_date = pd.NaT
            else:
                # Try various parsing methods
                try:
                    # Try standard pandas parsing first
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                    
                    # If that fails, try custom patterns
                    if pd.isna(parsed_date):
                        # Handle year-only format (e.g., "2020" -> "2020-01-01")
                        if re.match(r'^\d{4}$', date_str):
                            parsed_date = pd.Timestamp(f"{date_str}-01-01")
                        
                        # Handle year-month format (e.g., "2023-01" -> "2023-01-01")
                        elif re.match(r'^\d{4}-\d{1,2}$', date_str):
                            parsed_date = pd.Timestamp(f"{date_str}-01")
                        
                        # Handle month year format (e.g., "Jan 2023" -> "2023-01-01")
                        elif re.match(r'^[A-Za-z]{3}\s+\d{4}$', date_str):
                            parsed_date = pd.to_datetime(date_str, format='%b %Y', errors='coerce')
                            
                        # Handle slash format (e.g., "2023/01" -> "2023-01-01")
                        elif re.match(r'^\d{4}/\d{1,2}$', date_str):
                            year, month = date_str.split('/')
                            parsed_date = pd.Timestamp(f"{year}-{month.zfill(2)}-01")
                            
                except Exception:
                    parsed_date = pd.NaT
        
        parsed_dates.append(parsed_date)
    
    return pd.Series(parsed_dates)


def _convert_to_float(series: pd.Series) -> pd.Series:
    """
    Convert series values to float, handling various formats.
    
    Converts numeric strings to float and sets non-numeric values to NaN.
    
    Args:
        series: Pandas Series to convert.
        
    Returns:
        pd.Series: Series with float values.
    """
    converted_values = []
    
    for value in series:
        converted_value = np.nan
        
        if pd.notna(value):
            try:
                # Handle string values
                if isinstance(value, str):
                    value = value.strip()
                    if value and value.lower() not in ['', 'nan', 'none', 'null']:
                        converted_value = float(value)
                else:
                    # Handle numeric values
                    converted_value = float(value)
            except (ValueError, TypeError):
                converted_value = np.nan
        
        converted_values.append(converted_value)
    
    return pd.Series(converted_values, dtype='float64')


def _create_empty_standardized_dataframe() -> pd.DataFrame:
    """
    Create an empty DataFrame with standardized column structure.
    
    Returns:
        pd.DataFrame: Empty DataFrame with correct columns and dtypes.
    """
    columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
    df = pd.DataFrame(columns=columns)
    
    # Set appropriate dtypes
    df['date'] = pd.to_datetime(df['date'])
    for col in ['food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']:
        df[col] = df[col].astype('float64')
    
    return df