"""Module for downloading FAO Food Price Index data."""

import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests


def download_fao_fpi_data(
    url: str = "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data.xls",
    cache_dir: str = "cache"
) -> BytesIO:
    """
    Download FAO Food Price Index Excel data with caching and retry logic.
    
    Downloads the FAO FPI Excel file from the specified URL, saves it to a local
    cache directory with a timestamp, and returns the content as a BytesIO object.
    Implements retry logic with exponential backoff for network resilience.
    
    Args:
        url: URL to download the Excel file from. Defaults to FAO FPI data URL.
        cache_dir: Directory to save the cached file. Defaults to 'cache'.
        
    Returns:
        BytesIO: Excel file content ready for processing.
        
    Raises:
        requests.exceptions.Timeout: If all retry attempts timeout.
        requests.exceptions.HTTPError: If server returns an HTTP error.
        requests.exceptions.ConnectionError: If unable to establish connection.
        requests.exceptions.RequestException: For other request-related errors.
        
    Example:
        >>> excel_data = download_fao_fpi_data()
        >>> print(f"Downloaded {len(excel_data.getvalue())} bytes")
        
        >>> excel_data = download_fao_fpi_data(cache_dir="/tmp/fao_cache")
        >>> excel_data.seek(0)  # Reset to beginning for reading
    """
    max_retries = 3
    timeout_seconds = 30
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            
            # Save to cache with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_filename = f"fao_fpi_data_{timestamp}.xls"
            cache_path = Path(cache_dir) / cache_filename
            
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            return BytesIO(response.content)
            
        except requests.exceptions.HTTPError:
            # Don't retry on HTTP errors (4xx, 5xx)
            raise
            
        except (requests.exceptions.Timeout, 
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            last_exception = e
            
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            else:
                # Re-raise the last exception after all retries exhausted
                raise
    
    # This should never be reached, but just in case
    if last_exception:
        raise last_exception


def validate_excel_structure(excel_data: BytesIO) -> Tuple[bool, str]:
    """
    Validate that the Excel file contains the expected FAO FPI structure.
    
    Checks for the presence of required sheets ('Annual', 'Monthly') and 
    validates that the Annual sheet contains the expected column headers
    for food price index categories.
    
    Args:
        excel_data: BytesIO object containing Excel file content.
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if structure is valid, False otherwise
            - error_message: Empty string if valid, descriptive error if invalid
            
    Raises:
        Exception: Re-raises any pandas exceptions that occur during reading
        
    Example:
        >>> excel_data = download_fao_fpi_data()
        >>> is_valid, error = validate_excel_structure(excel_data)
        >>> if not is_valid:
        ...     print(f"Validation failed: {error}")
        ...
        >>> # For a valid file
        >>> is_valid, error = validate_excel_structure(valid_excel_data)
        >>> assert is_valid == True
        >>> assert error == ""
    """
    try:
        # Reset BytesIO position to beginning
        excel_data.seek(0)
        
        # Read Excel file to get sheet names (use openpyxl for xlsx files)
        excel_file = pd.ExcelFile(excel_data, engine='openpyxl')
        sheet_names = excel_file.sheet_names
        
        # Check for required sheets
        required_sheets = {'Annual', 'Monthly'}
        missing_sheets = required_sheets - set(sheet_names)
        
        if missing_sheets:
            return False, f"Missing required sheet(s): {', '.join(sorted(missing_sheets))}"
        
        # Read the Annual sheet to check column structure
        annual_df = pd.read_excel(excel_data, sheet_name='Annual', nrows=0, engine='openpyxl')  # Just get headers
        column_names = annual_df.columns.tolist()
        
        # Convert to lowercase for case-insensitive matching
        column_names_lower = [col.lower() for col in column_names]
        
        # Required columns (case-insensitive)
        required_columns = {
            'food price index', 'meat', 'dairy', 'cereals', 'oils', 'sugar'
        }
        
        # Check if each required column substring exists in any of the actual columns
        missing_columns = []
        for required_col in required_columns:
            found = any(required_col in col_name for col_name in column_names_lower)
            if not found:
                missing_columns.append(required_col.title())
        
        if missing_columns:
            return False, f"Annual sheet missing required column(s): {', '.join(missing_columns)}"
        
        return True, ""
        
    except pd.errors.EmptyDataError:
        return False, "Excel file appears to be empty"
    except ValueError as e:
        if any(phrase in str(e).lower() for phrase in [
            "excel file format", "not supported", "unsupported format", 
            "invalid file", "corrupt", "cannot be determined"
        ]):
            return False, "File is not a valid Excel format"
        return False, f"Excel file validation error: {str(e)}"
    except Exception as e:
        error_msg = str(e).lower()
        if any(phrase in error_msg for phrase in [
            "excel", "format", "corrupt", "invalid", "zip", "not supported"
        ]):
            return False, "File is not a valid Excel format"
        return False, f"Unexpected error reading Excel file: {str(e)}"