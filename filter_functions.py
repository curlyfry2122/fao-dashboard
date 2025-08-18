"""
Data filtering functions for FAO dashboard.

These functions provide various filtering capabilities for FAO food price index data,
including date range filtering, index selection, and data validation.
"""

from datetime import datetime, date
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def filter_by_date_range(
    df: pd.DataFrame, 
    start_date: Union[str, date, datetime], 
    end_date: Union[str, date, datetime],
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Args:
        df: DataFrame with date column
        start_date: Start date for filtering (inclusive)
        end_date: End date for filtering (inclusive)
        date_column: Name of the date column
        
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If date column doesn't exist or dates are invalid
        KeyError: If date_column not found in DataFrame
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame")
    
    if df.empty:
        return df.copy()
    
    # Convert inputs to datetime
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if start_dt > end_dt:
        raise ValueError("Start date must be before or equal to end date")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValueError(f"Cannot convert {date_column} to datetime: {e}")
    
    # Filter the DataFrame
    mask = (df[date_column] >= start_dt) & (df[date_column] <= end_dt)
    filtered_df = df[mask].copy()
    
    logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
    return filtered_df


def filter_by_indices(
    df: pd.DataFrame,
    selected_indices: List[str],
    preserve_date: bool = True
) -> pd.DataFrame:
    """
    Filter DataFrame to include only selected price indices.
    
    Args:
        df: DataFrame with price index columns
        selected_indices: List of column names to keep
        preserve_date: Whether to always keep the date column
        
    Returns:
        DataFrame with only selected columns
        
    Raises:
        ValueError: If no valid indices are selected
        KeyError: If selected indices don't exist in DataFrame
    """
    if df.empty:
        return df.copy()
    
    if not selected_indices:
        raise ValueError("At least one index must be selected")
    
    # Check if all selected indices exist
    missing_indices = [idx for idx in selected_indices if idx not in df.columns]
    if missing_indices:
        raise KeyError(f"Selected indices not found in DataFrame: {missing_indices}")
    
    # Build column list
    columns_to_keep = selected_indices.copy()
    
    # Always preserve date column if it exists and preserve_date is True
    if preserve_date and 'date' in df.columns and 'date' not in columns_to_keep:
        columns_to_keep.insert(0, 'date')
    
    filtered_df = df[columns_to_keep].copy()
    
    logger.info(f"Filtered to {len(columns_to_keep)} columns: {columns_to_keep}")
    return filtered_df


def filter_by_value_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inclusive: bool = True
) -> pd.DataFrame:
    """
    Filter DataFrame by value range in a specific column.
    
    Args:
        df: DataFrame to filter
        column: Column name to filter on
        min_value: Minimum value (inclusive/exclusive based on inclusive param)
        max_value: Maximum value (inclusive/exclusive based on inclusive param)
        inclusive: Whether range bounds are inclusive
        
    Returns:
        Filtered DataFrame
        
    Raises:
        KeyError: If column doesn't exist
        ValueError: If min_value > max_value
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    if df.empty:
        return df.copy()
    
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")
    
    # Create mask
    mask = pd.Series([True] * len(df), index=df.index)
    
    if min_value is not None:
        if inclusive:
            mask &= (df[column] >= min_value)
        else:
            mask &= (df[column] > min_value)
    
    if max_value is not None:
        if inclusive:
            mask &= (df[column] <= max_value)
        else:
            mask &= (df[column] < max_value)
    
    filtered_df = df[mask].copy()
    
    logger.info(f"Value range filter on '{column}': {len(df)} -> {len(filtered_df)} rows")
    return filtered_df


def validate_data_structure(
    df: pd.DataFrame,
    required_columns: List[str],
    date_column: str = 'date'
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has the required structure for FAO data.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        date_column: Name of the date column to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check date column specifically
    if date_column not in df.columns:
        issues.append(f"Date column '{date_column}' not found")
    else:
        # Validate date column
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                pd.to_datetime(df[date_column])
            except Exception:
                issues.append(f"Date column '{date_column}' contains invalid dates")
        
        # Check for null dates
        if df[date_column].isnull().any():
            null_count = df[date_column].isnull().sum()
            issues.append(f"Date column has {null_count} null values")
    
    # Check for numeric columns
    numeric_columns = [col for col in required_columns if col != date_column]
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    issues.append(f"Column '{col}' cannot be converted to numeric")
    
    # Check for duplicates
    if date_column in df.columns:
        duplicate_dates = df[date_column].duplicated().sum()
        if duplicate_dates > 0:
            issues.append(f"Found {duplicate_dates} duplicate dates")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def get_latest_data_points(
    df: pd.DataFrame,
    n_points: int = 1,
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Get the most recent n data points from the DataFrame.
    
    Args:
        df: DataFrame with date column
        n_points: Number of latest points to return
        date_column: Name of the date column
        
    Returns:
        DataFrame with the latest n points
        
    Raises:
        KeyError: If date column doesn't exist
        ValueError: If n_points is not positive
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found")
    
    if df.empty:
        return df.copy()
    
    # Sort by date and get the last n points
    sorted_df = df.sort_values(date_column, ascending=False)
    latest_points = sorted_df.head(n_points).copy()
    
    # Return in chronological order
    return latest_points.sort_values(date_column)


def calculate_data_completeness(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.Series:
    """
    Calculate data completeness percentage for each column.
    
    Args:
        df: DataFrame to analyze
        columns: Specific columns to analyze (default: all columns)
        
    Returns:
        Series with completeness percentages
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    if columns is None:
        columns = df.columns.tolist()
    
    completeness = {}
    for col in columns:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness[col] = (non_null_count / total_count) * 100
        else:
            completeness[col] = 0.0
    
    return pd.Series(completeness)


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers in a specific column using various methods.
    
    Args:
        df: DataFrame to analyze
        column: Column to analyze for outliers
        method: Method to use ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outlier information
        
    Raises:
        KeyError: If column doesn't exist
        ValueError: If method is not supported
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")
    
    if df.empty:
        return pd.DataFrame()
    
    supported_methods = ['iqr', 'zscore', 'modified_zscore']
    if method not in supported_methods:
        raise ValueError(f"Method must be one of: {supported_methods}")
    
    result_df = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        result_df['is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        # Calculate z-scores manually without scipy
        data = df[column].dropna()
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((df[column] - mean) / std)
        result_df['is_outlier'] = z_scores > threshold
    
    elif method == 'modified_zscore':
        median = df[column].median()
        mad = np.median(np.abs(df[column] - median))
        modified_z_scores = 0.6745 * (df[column] - median) / mad
        result_df['is_outlier'] = np.abs(modified_z_scores) > threshold
    
    return result_df[result_df['is_outlier']]