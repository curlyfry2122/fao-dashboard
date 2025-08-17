"""Module for downloading FAO Food Price Index data."""

import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import requests


def download_fao_fpi_data(
    url: str = "https://www.fao.org/media/docs/worldfoodsituationlibraries/default-document-library/food_price_indices_data_aug.xls?sfvrsn=63809b16_85",
    cache_dir: str = "cache",
    fallback_urls: Optional[List[str]] = None
) -> BytesIO:
    """
    Download FAO Food Price Index Excel data with enhanced resilience.
    
    Downloads the FAO FPI Excel file with automatic format detection, fallback URLs,
    improved headers for server compatibility, and robust retry logic with
    exponential backoff for network resilience.
    
    Args:
        url: Primary URL to download the Excel file from. Defaults to FAO FPI data URL.
        cache_dir: Directory to save the cached file. Defaults to 'cache'.
        fallback_urls: Optional list of alternative URLs to try if primary fails.
        
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
        
        >>> # With fallback URLs
        >>> fallbacks = ["https://www.fao.org/.../Food_price_indices_data.xlsx"]
        >>> excel_data = download_fao_fpi_data(fallback_urls=fallbacks)
    """
    max_retries = 5  # Increased retries for better resilience
    timeout_seconds = 30
    
    # Default fallback URLs if none provided
    if fallback_urls is None:
        fallback_urls = [
            # Alternative current FAO URL
            "https://www.fao.org/media/docs/worldfoodsituationlibraries/default-document-library/food_price_index_nominal_real_aug.xls?sfvrsn=78b6121d_39",
            # Legacy URLs (may be outdated but kept for fallback)
            "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data.xls",
            "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data.xlsx",
            # Alternative path structures
            "https://www.fao.org/3/food_price_indices_data.xls",
            "https://www.fao.org/3/food_price_indices_data.xlsx"
        ]
    
    # All URLs to try (primary + fallbacks)
    urls_to_try = [url] + fallback_urls
    
    # Enhanced headers for better server compatibility
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    last_exception = None
    successful_url = None
    
    # Try each URL with retry logic
    for current_url in urls_to_try:
        print(f"Trying URL: {current_url}")  # Debug info
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(current_url, headers=headers, timeout=timeout_seconds)
                response.raise_for_status()
                
                # Detect file format from response headers or URL
                content_type = response.headers.get('content-type', '').lower()
                file_extension = 'xlsx' if ('xlsx' in current_url or 'openxml' in content_type) else 'xls'
                
                # Save to cache with timestamp and correct extension
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cache_filename = f"fao_fpi_data_{timestamp}.{file_extension}"
                cache_path = Path(cache_dir) / cache_filename
                
                with open(cache_path, 'wb') as f:
                    f.write(response.content)
                
                successful_url = current_url
                print(f"✅ Successfully downloaded from: {successful_url}")
                return BytesIO(response.content)
                
            except requests.exceptions.HTTPError as e:
                print(f"❌ HTTP error for {current_url}: {e}")
                last_exception = e
                # Don't retry HTTP errors for this URL, try next URL
                break
                
            except (requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException) as e:
                last_exception = e
                print(f"⚠️ Network error for {current_url} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                    sleep_time = 2 ** attempt
                    print(f"⏳ Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"❌ All retry attempts failed for {current_url}")
                    # Try next URL
                    break
    
    # If we get here, all URLs and retries have failed
    error_msg = f"Failed to download FAO data from all URLs: {urls_to_try}"
    if last_exception:
        error_msg += f". Last error: {str(last_exception)}"
    print(f"❌ {error_msg}")
    raise requests.exceptions.RequestException(error_msg)


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
            return False, f"Unable to read Excel file with any supported engine. Last error: {str(last_exception)}"
        
        sheet_names = excel_file.sheet_names
        
        # Check for required sheets (handle both old and new naming conventions)
        required_sheets_old = {'Annual', 'Monthly'}
        required_sheets_new = {'Annual', 'Indices_Monthly'}
        
        # Check if we have either the old or new format
        has_old_format = required_sheets_old.issubset(set(sheet_names))
        has_new_format = required_sheets_new.issubset(set(sheet_names))
        
        if not (has_old_format or has_new_format):
            missing_sheets = required_sheets_old - set(sheet_names)
            alternative_msg = f"Expected either {required_sheets_old} (old format) or {required_sheets_new} (new format)"
            return False, f"Missing required sheet(s): {', '.join(sorted(missing_sheets))}. {alternative_msg}. Available: {sheet_names}"
        
        # Read the Annual sheet to check column structure with same engine
        excel_data.seek(0)  # Reset position
        engine_to_use = excel_file.engine.name if hasattr(excel_file.engine, 'name') else 'xlrd'
        
        # FAO data has headers in row 2, not row 0, so read a few rows and get row 2 as headers
        annual_df = pd.read_excel(excel_data, sheet_name='Annual', nrows=5, engine=engine_to_use, header=None)
        
        # Check if we have enough rows and use row 2 as column names
        if len(annual_df) >= 3:
            column_names_raw = annual_df.iloc[2].tolist()  # Row 2 contains the actual headers
            # Filter out NaN values and convert to strings
            column_names = [str(col) for col in column_names_raw if pd.notna(col)]
        else:
            return False, "Annual sheet does not have expected header structure"
        
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