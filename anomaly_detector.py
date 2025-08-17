"""
Anomaly detection module for FAO Food Price Index Dashboard.
Provides statistical anomaly detection using sigma bands and historical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import plotly.graph_objects as go


def calculate_sigma_bands(
    df: pd.DataFrame,
    column: str,
    window: int = 60,
    sigma: float = 2.0,
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling sigma bands for anomaly detection.
    
    Args:
        df: DataFrame with time series data
        column: Column name to analyze
        window: Rolling window size in periods
        sigma: Number of standard deviations for bands
        min_periods: Minimum periods required for calculation
    
    Returns:
        DataFrame with original data plus sigma bands
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = df.copy()
    
    # Calculate rolling statistics
    rolling_mean = df[column].rolling(window=window, min_periods=min_periods).mean()
    rolling_std = df[column].rolling(window=window, min_periods=min_periods).std()
    
    # Calculate sigma bands
    result_df[f'{column}_rolling_mean'] = rolling_mean
    result_df[f'{column}_rolling_std'] = rolling_std
    result_df[f'{column}_upper_band'] = rolling_mean + (sigma * rolling_std)
    result_df[f'{column}_lower_band'] = rolling_mean - (sigma * rolling_std)
    
    return result_df


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    window: int = 60,
    sigma: float = 2.0,
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Detect anomalies using sigma band method.
    
    Args:
        df: DataFrame with time series data
        column: Column name to analyze
        window: Rolling window size
        sigma: Number of standard deviations for detection
        min_periods: Minimum periods for calculation
    
    Returns:
        DataFrame with anomaly detection results
    """
    # Calculate sigma bands
    bands_df = calculate_sigma_bands(df, column, window, sigma, min_periods)
    
    # Detect anomalies
    upper_band = bands_df[f'{column}_upper_band']
    lower_band = bands_df[f'{column}_lower_band']
    values = bands_df[column]
    
    # Mark anomalies
    bands_df[f'{column}_is_anomaly'] = (values > upper_band) | (values < lower_band)
    bands_df[f'{column}_anomaly_type'] = np.where(
        values > upper_band, 'high',
        np.where(values < lower_band, 'low', 'normal')
    )
    
    # Calculate sigma level for anomalies
    rolling_mean = bands_df[f'{column}_rolling_mean']
    rolling_std = bands_df[f'{column}_rolling_std']
    bands_df[f'{column}_sigma_level'] = np.where(
        rolling_std > 0,
        np.abs(values - rolling_mean) / rolling_std,
        0
    )
    
    return bands_df


def get_anomaly_summary(
    df: pd.DataFrame,
    column: str,
    date_column: str = 'date'
) -> Dict[str, Any]:
    """
    Generate summary statistics for detected anomalies.
    
    Args:
        df: DataFrame with anomaly detection results
        column: Column name analyzed
        date_column: Name of date column
    
    Returns:
        Dictionary with anomaly summary statistics
    """
    anomaly_col = f'{column}_is_anomaly'
    type_col = f'{column}_anomaly_type'
    sigma_col = f'{column}_sigma_level'
    
    if anomaly_col not in df.columns:
        return {
            'total_anomalies': 0,
            'high_anomalies': 0,
            'low_anomalies': 0,
            'anomaly_rate': 0.0,
            'max_sigma_level': 0.0,
            'recent_anomalies': [],
            'period_start': None,
            'period_end': None
        }
    
    # Filter anomalies
    anomalies = df[df[anomaly_col]].copy()
    
    # Basic counts
    total_anomalies = len(anomalies)
    high_anomalies = len(anomalies[anomalies[type_col] == 'high'])
    low_anomalies = len(anomalies[anomalies[type_col] == 'low'])
    total_points = len(df)
    anomaly_rate = (total_anomalies / total_points * 100) if total_points > 0 else 0
    
    # Sigma levels
    max_sigma_level = anomalies[sigma_col].max() if not anomalies.empty else 0
    
    # Recent anomalies (last 30 days)
    if not anomalies.empty and date_column in df.columns:
        recent_cutoff = df[date_column].max() - timedelta(days=30)
        recent_anomalies = anomalies[anomalies[date_column] >= recent_cutoff]
        recent_list = []
        
        for _, row in recent_anomalies.head(10).iterrows():
            recent_list.append({
                'date': row[date_column],
                'value': row[column],
                'type': row[type_col],
                'sigma_level': row[sigma_col],
                'rolling_mean': row[f'{column}_rolling_mean']
            })
    else:
        recent_list = []
    
    # Date range
    period_start = df[date_column].min() if date_column in df.columns else None
    period_end = df[date_column].max() if date_column in df.columns else None
    
    return {
        'total_anomalies': total_anomalies,
        'high_anomalies': high_anomalies,
        'low_anomalies': low_anomalies,
        'anomaly_rate': anomaly_rate,
        'max_sigma_level': max_sigma_level,
        'recent_anomalies': recent_list,
        'period_start': period_start,
        'period_end': period_end
    }


def classify_anomaly_severity(sigma_level: float) -> str:
    """
    Classify anomaly severity based on sigma level.
    
    Args:
        sigma_level: Number of standard deviations from mean
    
    Returns:
        Severity classification string
    """
    if sigma_level >= 3.0:
        return "Extreme"
    elif sigma_level >= 2.5:
        return "Severe"
    elif sigma_level >= 2.0:
        return "Moderate"
    elif sigma_level >= 1.5:
        return "Mild"
    else:
        return "Weak"


def add_anomaly_visualization(
    fig: go.Figure,
    df: pd.DataFrame,
    column: str,
    show_bands: bool = True,
    show_anomalies: bool = True,
    high_color: str = '#FF4444',
    low_color: str = '#4444FF',
    band_color: str = 'rgba(128, 128, 128, 0.2)'
) -> go.Figure:
    """
    Add anomaly visualization to existing Plotly figure.
    
    Args:
        fig: Existing Plotly figure
        df: DataFrame with anomaly detection results
        column: Column name being visualized
        show_bands: Whether to show sigma bands
        show_anomalies: Whether to show anomaly markers
        high_color: Color for high anomalies
        low_color: Color for low anomalies
        band_color: Color for sigma bands
    
    Returns:
        Enhanced Plotly figure with anomaly visualization
    """
    if f'{column}_upper_band' not in df.columns:
        return fig
    
    x_values = df.index
    
    # Add sigma bands
    if show_bands:
        # Upper band
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[f'{column}_upper_band'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name=f'{column}_upper_band'
        ))
        
        # Lower band
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[f'{column}_lower_band'],
            mode='lines',
            fill='tonexty',
            fillcolor=band_color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name=f'{column}_lower_band'
        ))
        
        # Add band lines (dashed)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[f'{column}_upper_band'],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip',
            name='Upper Band'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[f'{column}_lower_band'],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip',
            name='Lower Band'
        ))
    
    # Add anomaly markers
    if show_anomalies and f'{column}_is_anomaly' in df.columns:
        anomalies = df[df[f'{column}_is_anomaly']].copy()
        
        if not anomalies.empty:
            # High anomalies
            high_anomalies = anomalies[anomalies[f'{column}_anomaly_type'] == 'high']
            if not high_anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=high_anomalies.index,
                    y=high_anomalies[column],
                    mode='markers',
                    marker=dict(
                        color=high_color,
                        size=8,
                        symbol='triangle-up',
                        line=dict(width=1, color='white')
                    ),
                    name='High Anomalies',
                    hovertemplate='<b>High Anomaly</b><br>' +
                                  'Date: %{x}<br>' +
                                  'Value: %{y:.1f}<br>' +
                                  'Sigma Level: %{customdata:.1f}σ<extra></extra>',
                    customdata=high_anomalies[f'{column}_sigma_level']
                ))
            
            # Low anomalies
            low_anomalies = anomalies[anomalies[f'{column}_anomaly_type'] == 'low']
            if not low_anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=low_anomalies.index,
                    y=low_anomalies[column],
                    mode='markers',
                    marker=dict(
                        color=low_color,
                        size=8,
                        symbol='triangle-down',
                        line=dict(width=1, color='white')
                    ),
                    name='Low Anomalies',
                    hovertemplate='<b>Low Anomaly</b><br>' +
                                  'Date: %{x}<br>' +
                                  'Value: %{y:.1f}<br>' +
                                  'Sigma Level: %{customdata:.1f}σ<extra></extra>',
                    customdata=low_anomalies[f'{column}_sigma_level']
                ))
    
    return fig


def identify_historical_periods(
    df: pd.DataFrame,
    date_column: str = 'date'
) -> List[Dict[str, Any]]:
    """
    Identify known historical food crisis periods.
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column
    
    Returns:
        List of dictionaries describing historical periods
    """
    historical_periods = [
        {
            'name': '2007-2008 Food Crisis',
            'start_date': '2007-01-01',
            'end_date': '2008-12-31',
            'description': 'Global food price crisis with dramatic increases in wheat, rice, and corn prices',
            'color': 'rgba(255, 0, 0, 0.3)'
        },
        {
            'name': '2010-2011 Price Volatility',
            'start_date': '2010-06-01',
            'end_date': '2011-09-30',
            'description': 'Russian grain export ban and weather-related supply disruptions',
            'color': 'rgba(255, 165, 0, 0.3)'
        },
        {
            'name': 'COVID-19 Pandemic Impact',
            'start_date': '2020-03-01',
            'end_date': '2020-12-31',
            'description': 'Supply chain disruptions and trade restrictions due to pandemic',
            'color': 'rgba(128, 0, 128, 0.3)'
        },
        {
            'name': 'Ukraine Crisis',
            'start_date': '2022-02-01',
            'end_date': '2023-12-31',
            'description': 'Grain and fertilizer supply disruptions due to conflict',
            'color': 'rgba(255, 255, 0, 0.3)'
        }
    ]
    
    # Filter periods that overlap with data range
    if date_column not in df.columns or df.empty:
        return []
    
    data_start = df[date_column].min()
    data_end = df[date_column].max()
    
    relevant_periods = []
    for period in historical_periods:
        period_start = pd.to_datetime(period['start_date'])
        period_end = pd.to_datetime(period['end_date'])
        
        # Check if period overlaps with data range
        if period_start <= data_end and period_end >= data_start:
            # Adjust dates to data range
            adjusted_start = max(period_start, data_start)
            adjusted_end = min(period_end, data_end)
            
            period_copy = period.copy()
            period_copy['adjusted_start'] = adjusted_start
            period_copy['adjusted_end'] = adjusted_end
            relevant_periods.append(period_copy)
    
    return relevant_periods


def add_historical_period_shading(
    fig: go.Figure,
    periods: List[Dict[str, Any]]
) -> go.Figure:
    """
    Add shading for historical crisis periods to chart.
    
    Args:
        fig: Plotly figure to enhance
        periods: List of historical periods with date ranges
    
    Returns:
        Enhanced figure with period shading
    """
    for period in periods:
        if 'adjusted_start' in period and 'adjusted_end' in period:
            fig.add_vrect(
                x0=period['adjusted_start'],
                x1=period['adjusted_end'],
                fillcolor=period['color'],
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text=period['name'],
                annotation_position="top left",
                annotation=dict(
                    font=dict(size=10, color="gray"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
    
    return fig


def calculate_anomaly_frequency(
    df: pd.DataFrame,
    column: str,
    period: str = 'M'
) -> pd.DataFrame:
    """
    Calculate anomaly frequency over time periods.
    
    Args:
        df: DataFrame with anomaly detection results
        column: Column name analyzed
        period: Frequency period ('D', 'W', 'M', 'Q', 'Y')
    
    Returns:
        DataFrame with anomaly frequency by period
    """
    anomaly_col = f'{column}_is_anomaly'
    
    if anomaly_col not in df.columns or 'date' not in df.columns:
        return pd.DataFrame()
    
    # Group by period and count anomalies
    df_copy = df.copy()
    df_copy['period'] = df_copy['date'].dt.to_period(period)
    
    frequency_df = df_copy.groupby('period').agg({
        anomaly_col: ['sum', 'count'],
        f'{column}_anomaly_type': lambda x: (x == 'high').sum(),
        column: ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    frequency_df.columns = [
        'total_anomalies', 'total_points', 'high_anomalies', 
        'mean_value', 'std_value'
    ]
    
    # Calculate rates
    frequency_df['anomaly_rate'] = (
        frequency_df['total_anomalies'] / frequency_df['total_points'] * 100
    ).round(1)
    frequency_df['low_anomalies'] = (
        frequency_df['total_anomalies'] - frequency_df['high_anomalies']
    )
    
    return frequency_df.reset_index()


def get_anomaly_insights(
    df: pd.DataFrame,
    column: str,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Generate textual insights about detected anomalies.
    
    Args:
        df: DataFrame with anomaly detection results
        column: Column name analyzed
        top_n: Number of top anomalies to highlight
    
    Returns:
        Dictionary with anomaly insights and recommendations
    """
    summary = get_anomaly_summary(df, column)
    
    if summary['total_anomalies'] == 0:
        return {
            'overall_assessment': 'No significant anomalies detected in the selected period.',
            'top_anomalies': [],
            'recommendations': ['Data appears stable with no unusual price movements.'],
            'trend_analysis': 'Normal price variation within expected statistical bounds.'
        }
    
    # Overall assessment
    rate = summary['anomaly_rate']
    if rate > 10:
        assessment = f"High anomaly rate ({rate:.1f}%) indicates significant price volatility."
    elif rate > 5:
        assessment = f"Moderate anomaly rate ({rate:.1f}%) suggests some price instability."
    else:
        assessment = f"Low anomaly rate ({rate:.1f}%) indicates relatively stable prices."
    
    # Top anomalies
    anomaly_col = f'{column}_is_anomaly'
    sigma_col = f'{column}_sigma_level'
    
    if anomaly_col in df.columns:
        top_anomalies = df[df[anomaly_col]].nlargest(top_n, sigma_col)
        top_list = []
        
        for _, row in top_anomalies.iterrows():
            severity = classify_anomaly_severity(row[sigma_col])
            top_list.append({
                'date': row.get('date', 'Unknown'),
                'value': row[column],
                'sigma_level': row[sigma_col],
                'severity': severity,
                'type': row[f'{column}_anomaly_type']
            })
    else:
        top_list = []
    
    # Recommendations
    recommendations = []
    if summary['high_anomalies'] > summary['low_anomalies']:
        recommendations.append("Monitor for supply disruptions causing price spikes.")
    elif summary['low_anomalies'] > summary['high_anomalies']:
        recommendations.append("Investigate potential oversupply or demand reduction.")
    
    if summary['max_sigma_level'] > 3:
        recommendations.append("Extreme price movements detected - investigate underlying causes.")
    
    if summary['anomaly_rate'] > 10:
        recommendations.append("High volatility suggests market instability - consider risk management measures.")
    
    if not recommendations:
        recommendations.append("Continue monitoring for emerging price trends.")
    
    # Trend analysis
    if summary['high_anomalies'] > 0 and summary['low_anomalies'] > 0:
        trend = "Mixed anomaly pattern with both upward and downward price spikes."
    elif summary['high_anomalies'] > 0:
        trend = "Predominantly upward price anomalies indicating inflationary pressures."
    elif summary['low_anomalies'] > 0:
        trend = "Predominantly downward price anomalies indicating deflationary pressures."
    else:
        trend = "No clear anomaly trend detected."
    
    return {
        'overall_assessment': assessment,
        'top_anomalies': top_list,
        'recommendations': recommendations,
        'trend_analysis': trend
    }