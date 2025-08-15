"""Module for calculating derived metrics from FAO Food Price Index data."""

from typing import List

import numpy as np
import pandas as pd


def calculate_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Calculate derived metrics from FAO Food Price Index data.
    
    Adds calculated columns to the input DataFrame based on the requested metrics.
    All calculations use pandas vectorized operations for optimal performance.
    
    Args:
        df: DataFrame with FAO data containing columns:
            ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        metrics: List of metrics to calculate. Supported values:
            - 'yoy_change': Year-over-year percentage change
            - 'mom_change': Month-over-month percentage change  
            - '12m_avg': 12-month rolling average
            
    Returns:
        pd.DataFrame: Original DataFrame with additional calculated columns.
        New columns follow naming pattern: {original_column}_{metric}
        
    Raises:
        ValueError: If unknown metric is requested or required columns are missing.
        
    Example:
        >>> from excel_parser import parse_fao_excel_data
        >>> df = parse_fao_excel_data(excel_data, 'Monthly')
        >>> enriched_df = calculate_metrics(df, ['yoy_change', 'mom_change'])
        >>> print(enriched_df.columns.tolist())
        ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar',
         'food_price_index_yoy_change', 'food_price_index_mom_change', ...]
    """
    # Validate input DataFrame
    _validate_input_dataframe(df)
    
    # Validate requested metrics
    valid_metrics = {'yoy_change', 'mom_change', '12m_avg'}
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Unknown metric(s): {', '.join(invalid_metrics)}. "
                        f"Valid metrics are: {', '.join(valid_metrics)}")
    
    # Return original DataFrame if no metrics requested
    if not metrics:
        return df.copy()
    
    # Create a copy to avoid modifying original DataFrame
    result_df = df.copy()
    
    # Sort by date to ensure proper time series calculations
    result_df = result_df.sort_values('date').reset_index(drop=True)
    
    # Get numeric columns (excluding date)
    numeric_columns = [col for col in result_df.columns if col != 'date']
    
    # Calculate each requested metric for all numeric columns
    for metric in metrics:
        if metric == 'yoy_change':
            result_df = _calculate_yoy_change(result_df, numeric_columns)
        elif metric == 'mom_change':
            result_df = _calculate_mom_change(result_df, numeric_columns)
        elif metric == '12m_avg':
            result_df = _calculate_12m_avg(result_df, numeric_columns)
    
    return result_df


def _validate_input_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the input DataFrame has the required structure.
    
    Args:
        df: DataFrame to validate.
        
    Raises:
        ValueError: If DataFrame doesn't have required columns or structure.
    """
    required_columns = ['date', 'food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
    
    # Skip date type validation for empty DataFrames
    if len(df) > 0 and not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("'date' column must be datetime type")


def _calculate_yoy_change(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Calculate year-over-year percentage change for numeric columns.
    
    Automatically detects data frequency and uses appropriate shift:
    - Annual data: 1-period shift
    - Monthly data: 12-period shift
    Formula: ((current - previous_year) / previous_year) * 100
    
    Args:
        df: DataFrame with time series data sorted by date.
        numeric_columns: List of column names to calculate YoY change for.
        
    Returns:
        pd.DataFrame: DataFrame with added YoY change columns.
    """
    result_df = df.copy()
    
    # Detect frequency to determine shift periods
    shift_periods = _detect_yoy_shift_periods(result_df)
    
    for col in numeric_columns:
        # Calculate YoY change using appropriate shift
        previous_year_values = result_df[col].shift(shift_periods)
        yoy_change = ((result_df[col] - previous_year_values) / previous_year_values) * 100
        
        result_df[f'{col}_yoy_change'] = yoy_change
    
    return result_df


def _calculate_mom_change(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Calculate month-over-month percentage change for numeric columns.
    
    Uses pandas pct_change() for efficient calculation.
    Formula: ((current - previous) / previous) * 100
    
    Args:
        df: DataFrame with time series data sorted by date.
        numeric_columns: List of column names to calculate MoM change for.
        
    Returns:
        pd.DataFrame: DataFrame with added MoM change columns.
    """
    result_df = df.copy()
    
    for col in numeric_columns:
        # Calculate MoM change using pct_change and convert to percentage
        # Use fill_method=None to avoid deprecation warning
        mom_change = result_df[col].pct_change(fill_method=None) * 100
        
        result_df[f'{col}_mom_change'] = mom_change
    
    return result_df


def _calculate_12m_avg(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Calculate 12-month rolling average for numeric columns.
    
    Uses pandas rolling window operation for efficient calculation.
    
    Args:
        df: DataFrame with time series data sorted by date.
        numeric_columns: List of column names to calculate 12m average for.
        
    Returns:
        pd.DataFrame: DataFrame with added 12m average columns.
    """
    result_df = df.copy()
    
    for col in numeric_columns:
        # Calculate 12-month rolling average
        # min_periods=12 ensures we only get values when we have a full 12 months
        rolling_avg = result_df[col].rolling(window=12, min_periods=12).mean()
        
        result_df[f'{col}_12m_avg'] = rolling_avg
    
    return result_df


def _detect_yoy_shift_periods(df: pd.DataFrame) -> int:
    """
    Detect the appropriate number of periods to shift for YoY calculation.
    
    Analyzes the date frequency to determine if data is annual, monthly, etc.
    
    Args:
        df: DataFrame with 'date' column.
        
    Returns:
        int: Number of periods to shift (1 for annual, 12 for monthly).
    """
    if len(df) < 2:
        return 12  # Default to monthly
    
    # Calculate the average time difference between consecutive dates
    date_diffs = df['date'].diff().dropna()
    
    if len(date_diffs) == 0:
        return 12  # Default to monthly
    
    avg_diff = date_diffs.mean()
    
    # If average difference is close to 1 year (around 365 days), it's annual data
    if avg_diff >= pd.Timedelta(days=300):
        return 1  # Annual data - compare to previous year (1 period back)
    else:
        return 12  # Monthly or other frequent data - compare to 12 periods back