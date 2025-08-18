"""
KPI Calculator for FAO Food Price Index Dashboard.

Calculates key performance indicators from filtered FAO data including
current values, 12-month averages, year-over-year changes, and trend direction.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from performance_monitor import performance_monitor, performance_context

logger = logging.getLogger(__name__)


@performance_monitor('kpi_calculation', include_args=True)
def calculate_kpis(
    df: pd.DataFrame, 
    selected_indices: List[str], 
    date_column: str = 'date'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate key performance indicators for selected indices.
    
    Args:
        df: DataFrame with FAO data containing date and index columns
        selected_indices: List of column names to calculate KPIs for
        date_column: Name of the date column
        
    Returns:
        Dictionary with structure:
        {
            'column_name': {
                'current_value': float,
                '12m_avg': float,
                'yoy_change': float,
                'trend_direction': str  # 'up', 'down', 'stable'
            }
        }
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
        KeyError: If selected indices don't exist in DataFrame
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame")
    
    # Validate selected indices exist
    missing_indices = [idx for idx in selected_indices if idx not in df.columns]
    if missing_indices:
        raise KeyError(f"Selected indices not found in DataFrame: {missing_indices}")
    
    # Sort by date to ensure proper time series calculations
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    
    kpis = {}
    
    for index in selected_indices:
        try:
            kpis[index] = {
                'current_value': _get_current_value(df_sorted, index),
                '12m_avg': _calculate_12m_average(df_sorted, index),
                'yoy_change': _calculate_yoy_change(df_sorted, index),
                'trend_direction': _determine_trend_direction(df_sorted, index)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate KPIs for {index}: {e}")
            kpis[index] = {
                'current_value': None,
                '12m_avg': None,
                'yoy_change': None,
                'trend_direction': 'unknown'
            }
    
    return kpis


def _get_current_value(df: pd.DataFrame, column: str) -> Optional[float]:
    """Get the most recent non-null value for a column."""
    if df.empty or column not in df.columns:
        return None
    
    # Get last non-null value
    non_null_values = df[column].dropna()
    if non_null_values.empty:
        return None
    
    return float(non_null_values.iloc[-1])


def _calculate_12m_average(df: pd.DataFrame, column: str) -> Optional[float]:
    """Calculate 12-month rolling average for the latest available period."""
    if df.empty or column not in df.columns:
        return None
    
    # Calculate 12-month rolling average
    rolling_avg = df[column].rolling(window=12, min_periods=12).mean()
    
    # Get the most recent non-null rolling average
    non_null_avg = rolling_avg.dropna()
    if non_null_avg.empty:
        return None
    
    return float(non_null_avg.iloc[-1])


def _calculate_yoy_change(df: pd.DataFrame, column: str) -> Optional[float]:
    """Calculate year-over-year percentage change for the latest period."""
    if df.empty or column not in df.columns or len(df) < 12:
        return None
    
    try:
        # Get current and previous year values (12 periods back for monthly data)
        current_value = df[column].iloc[-1]
        previous_year_value = df[column].iloc[-13] if len(df) >= 13 else df[column].iloc[-12]
        
        if pd.isna(current_value) or pd.isna(previous_year_value) or previous_year_value == 0:
            return None
        
        yoy_change = ((current_value - previous_year_value) / previous_year_value) * 100
        return float(yoy_change)
    
    except (IndexError, ZeroDivisionError):
        return None


def _determine_trend_direction(df: pd.DataFrame, column: str, periods: int = 3) -> str:
    """
    Determine trend direction based on recent data points.
    
    Args:
        df: DataFrame with time series data
        column: Column to analyze
        periods: Number of recent periods to analyze (default: 3)
        
    Returns:
        'up', 'down', 'stable', or 'unknown'
    """
    if df.empty or column not in df.columns or len(df) < periods:
        return 'unknown'
    
    try:
        # Get the last 'periods' non-null values
        recent_values = df[column].dropna().tail(periods)
        
        if len(recent_values) < periods:
            return 'unknown'
        
        recent_values_list = recent_values.tolist()
        
        # Calculate trend using linear regression slope
        x = np.arange(len(recent_values_list))
        slope = np.polyfit(x, recent_values_list, 1)[0]
        
        # Determine threshold for "stable" (0.5% of the mean value)
        mean_value = np.mean(recent_values_list)
        threshold = abs(mean_value * 0.005) if mean_value != 0 else 0.1
        
        if slope > threshold:
            return 'up'
        elif slope < -threshold:
            return 'down'
        else:
            return 'stable'
    
    except Exception as e:
        logger.warning(f"Failed to determine trend for {column}: {e}")
        return 'unknown'


def get_kpi_summary(kpis: Dict[str, Dict[str, float]]) -> Dict[str, any]:
    """
    Generate summary statistics across all KPIs.
    
    Args:
        kpis: Dictionary of KPIs from calculate_kpis()
        
    Returns:
        Dictionary with summary statistics
    """
    if not kpis:
        return {
            'total_indices': 0,
            'avg_yoy_change': None,
            'indices_trending_up': 0,
            'indices_trending_down': 0,
            'indices_stable': 0
        }
    
    yoy_changes = []
    trend_counts = {'up': 0, 'down': 0, 'stable': 0, 'unknown': 0}
    
    for index_name, metrics in kpis.items():
        if metrics['yoy_change'] is not None:
            yoy_changes.append(metrics['yoy_change'])
        
        trend = metrics.get('trend_direction', 'unknown')
        trend_counts[trend] += 1
    
    avg_yoy = np.mean(yoy_changes) if yoy_changes else None
    
    return {
        'total_indices': len(kpis),
        'avg_yoy_change': float(avg_yoy) if avg_yoy is not None else None,
        'indices_trending_up': trend_counts['up'],
        'indices_trending_down': trend_counts['down'],
        'indices_stable': trend_counts['stable'],
        'indices_unknown_trend': trend_counts['unknown']
    }


def format_kpi_for_display(
    value: Optional[float], 
    metric_type: str = 'value',
    decimal_places: int = 1
) -> str:
    """
    Format KPI values for display in Streamlit metrics.
    
    Args:
        value: The numeric value to format
        metric_type: Type of metric ('value', 'percentage', 'change')
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted string for display
    """
    if value is None:
        return "N/A"
    
    if metric_type == 'percentage' or metric_type == 'change':
        return f"{value:.{decimal_places}f}%"
    else:
        return f"{value:.{decimal_places}f}"


def get_trend_emoji(trend_direction: str) -> str:
    """
    Get emoji representation for trend direction.
    
    Args:
        trend_direction: 'up', 'down', 'stable', or 'unknown'
        
    Returns:
        Emoji string
    """
    trend_emojis = {
        'up': '📈',
        'down': '📉', 
        'stable': '➡️',
        'unknown': '❓'
    }
    
    return trend_emojis.get(trend_direction, '❓')


def get_delta_color(yoy_change: Optional[float]) -> Optional[str]:
    """
    Get delta color for st.metric based on year-over-year change.
    
    Args:
        yoy_change: Year-over-year percentage change
        
    Returns:
        'normal', 'inverse', or None
    """
    if yoy_change is None:
        return None
    
    # For food price indices, increases are generally considered negative (inflation)
    # so we use 'inverse' to show increases in red and decreases in green
    return 'inverse'


def calculate_detailed_statistics(
    df: pd.DataFrame,
    selected_indices: List[str],
    index_mapping: Dict[str, str],
    sparkline_periods: int = 24
) -> pd.DataFrame:
    """
    Calculate detailed statistics for selected indices including sparkline data.
    
    Args:
        df: DataFrame with FAO data
        selected_indices: List of display names for selected indices
        index_mapping: Mapping from display names to column names
        sparkline_periods: Number of recent periods for sparkline (default: 24)
        
    Returns:
        DataFrame with columns: Index, Min, Max, Mean, Std Dev, Sparkline
        
    Raises:
        ValueError: If DataFrame is empty or no valid indices selected
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Get column names from display names
    selected_columns = [index_mapping[idx] for idx in selected_indices 
                       if idx in index_mapping and index_mapping[idx] in df.columns]
    
    if not selected_columns:
        raise ValueError("No valid indices selected")
    
    # Sort by date to ensure proper time series
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    statistics_data = []
    
    for display_name in selected_indices:
        if display_name in index_mapping:
            column_name = index_mapping[display_name]
            
            if column_name in df_sorted.columns:
                # Calculate basic statistics
                series = df_sorted[column_name].dropna()
                
                if len(series) > 0:
                    min_val = float(series.min())
                    max_val = float(series.max())
                    mean_val = float(series.mean())
                    std_val = float(series.std())
                    
                    # Prepare sparkline data (last N periods)
                    sparkline_data = series.tail(sparkline_periods).tolist()
                    
                    statistics_data.append({
                        'Index': display_name,
                        'Min': min_val,
                        'Max': max_val,
                        'Mean': mean_val,
                        'Std Dev': std_val,
                        'Sparkline': sparkline_data
                    })
    
    if not statistics_data:
        # Return empty DataFrame with correct structure if no data
        return pd.DataFrame(columns=['Index', 'Min', 'Max', 'Mean', 'Std Dev', 'Sparkline'])
    
    return pd.DataFrame(statistics_data)