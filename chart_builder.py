"""Module for building interactive charts using Plotly."""

import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
from anomaly_detector import (
    detect_anomalies,
    add_anomaly_visualization,
    identify_historical_periods,
    add_historical_period_shading
)


def build_chart(
    df: pd.DataFrame, 
    chart_type: str, 
    indices: List[str],
    anomaly_detection: Optional[Dict] = None
) -> go.Figure:
    """
    Build a Plotly figure based on chart type and selected indices.
    
    Args:
        df: DataFrame with date index and price index columns
        chart_type: Type of chart ('Line Chart', 'Area Chart', 'Year-over-Year Change')
        indices: List of column names to plot
        anomaly_detection: Optional dict with anomaly detection settings
                          {'enabled': bool, 'sigma': float, 'window': int, 'show_bands': bool,
                           'show_historical': bool, 'high_color': str, 'low_color': str}
        
    Returns:
        go.Figure: Plotly figure object ready for display
    """
    if chart_type == 'Line Chart':
        return _build_line_chart(df, indices, anomaly_detection)
    elif chart_type == 'Area Chart':
        return _build_area_chart(df, indices, anomaly_detection)
    elif chart_type == 'Year-over-Year Change':
        return _build_yoy_chart(df, indices, anomaly_detection)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")


def _build_line_chart(df: pd.DataFrame, indices: List[str], anomaly_detection: Optional[Dict] = None) -> go.Figure:
    """Build a multi-line chart for selected indices."""
    fig = go.Figure()
    
    # Define colors for consistent styling
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, idx in enumerate(indices):
        if idx in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[idx],
                mode='lines',
                name=idx,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='%{x|%b %Y}<br>%{y:.1f}<extra></extra>'
            ))
    
    # Add anomaly detection if enabled
    if anomaly_detection and anomaly_detection.get('enabled', False):
        df_with_anomalies = df.copy()
        
        # Add historical period shading if requested
        if anomaly_detection.get('show_historical', False):
            historical_periods = identify_historical_periods(df_with_anomalies)
            fig = add_historical_period_shading(fig, historical_periods)
        
        # Process each index for anomaly detection
        for idx in indices:
            if idx in df.columns:
                # Detect anomalies
                df_with_anomalies = detect_anomalies(
                    df_with_anomalies, 
                    idx,
                    window=anomaly_detection.get('window', 60),
                    sigma=anomaly_detection.get('sigma', 2.0)
                )
                
                # Add anomaly visualization
                fig = add_anomaly_visualization(
                    fig,
                    df_with_anomalies,
                    idx,
                    show_bands=anomaly_detection.get('show_bands', True),
                    show_anomalies=True,
                    high_color=anomaly_detection.get('high_color', '#FF4444'),
                    low_color=anomaly_detection.get('low_color', '#4444FF')
                )
    
    fig.update_layout(
        title='Food Price Indices Trend',
        xaxis_title='Date',
        yaxis_title='Index Value',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=500 if anomaly_detection and anomaly_detection.get('enabled') else 400
    )
    
    return fig


def _build_area_chart(df: pd.DataFrame, indices: List[str], anomaly_detection: Optional[Dict] = None) -> go.Figure:
    """Build a stacked area chart for selected indices."""
    fig = go.Figure()
    
    # Define colors with transparency for area chart
    colors = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 
              'rgba(44, 160, 44, 0.8)', 'rgba(214, 39, 40, 0.8)',
              'rgba(148, 103, 189, 0.8)', 'rgba(140, 86, 75, 0.8)']
    
    for i, idx in enumerate(indices):
        if idx in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[idx],
                mode='lines',
                name=idx,
                fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(width=0.5),
                fillcolor=colors[i % len(colors)],
                hovertemplate='%{x|%b %Y}<br>%{y:.1f}<extra></extra>'
            ))
    
    # Add anomaly detection if enabled (markers only for area charts)
    if anomaly_detection and anomaly_detection.get('enabled', False):
        df_with_anomalies = df.copy()
        
        # Add historical period shading if requested
        if anomaly_detection.get('show_historical', False):
            historical_periods = identify_historical_periods(df_with_anomalies)
            fig = add_historical_period_shading(fig, historical_periods)
        
        # Process each index for anomaly detection (markers only, no bands)
        for idx in indices:
            if idx in df.columns:
                # Detect anomalies
                df_with_anomalies = detect_anomalies(
                    df_with_anomalies, 
                    idx,
                    window=anomaly_detection.get('window', 60),
                    sigma=anomaly_detection.get('sigma', 2.0)
                )
                
                # Add anomaly markers only (no bands for area charts)
                fig = add_anomaly_visualization(
                    fig,
                    df_with_anomalies,
                    idx,
                    show_bands=False,  # Don't show bands on area charts
                    show_anomalies=True,
                    high_color=anomaly_detection.get('high_color', '#FF4444'),
                    low_color=anomaly_detection.get('low_color', '#4444FF')
                )
    
    fig.update_layout(
        title='Food Price Indices Composition',
        xaxis_title='Date',
        yaxis_title='Index Value',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=500 if anomaly_detection and anomaly_detection.get('enabled') else 400
    )
    
    return fig


def _build_yoy_chart(df: pd.DataFrame, indices: List[str], anomaly_detection: Optional[Dict] = None) -> go.Figure:
    """Build a year-over-year percentage change chart."""
    fig = go.Figure()
    
    # Define colors for consistent styling
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, idx in enumerate(indices):
        # Check if YoY column exists, otherwise calculate it
        yoy_col = f"{idx}_yoy_change"
        
        if yoy_col in df.columns:
            y_data = df[yoy_col]
        elif idx in df.columns:
            # Calculate YoY change if not present
            y_data = df[idx].pct_change(periods=12) * 100
        else:
            continue
            
        fig.add_trace(go.Scatter(
            x=df.index,
            y=y_data,
            mode='lines',
            name=f"{idx} (YoY %)",
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='%{x|%b %Y}<br>%{y:.1f}%<extra></extra>'
        ))
    
    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add anomaly detection for YoY data if enabled
    if anomaly_detection and anomaly_detection.get('enabled', False):
        df_with_yoy = df.copy()
        
        # Add historical period shading if requested
        if anomaly_detection.get('show_historical', False):
            historical_periods = identify_historical_periods(df_with_yoy)
            fig = add_historical_period_shading(fig, historical_periods)
        
        # Process each index for YoY anomaly detection
        for idx in indices:
            yoy_col = f"{idx}_yoy_change"
            
            if yoy_col in df.columns:
                temp_col = yoy_col
            elif idx in df.columns:
                # Calculate YoY and add to dataframe
                df_with_yoy[f'{idx}_yoy_temp'] = df[idx].pct_change(periods=12) * 100
                temp_col = f'{idx}_yoy_temp'
            else:
                continue
            
            # Detect anomalies on YoY data
            df_with_yoy = detect_anomalies(
                df_with_yoy, 
                temp_col,
                window=anomaly_detection.get('window', 60),
                sigma=anomaly_detection.get('sigma', 2.0)
            )
            
            # Add anomaly visualization
            fig = add_anomaly_visualization(
                fig,
                df_with_yoy,
                temp_col,
                show_bands=anomaly_detection.get('show_bands', True),
                show_anomalies=True,
                high_color=anomaly_detection.get('high_color', '#FF4444'),
                low_color=anomaly_detection.get('low_color', '#4444FF')
            )
    
    fig.update_layout(
        title='Year-over-Year Percentage Change',
        xaxis_title='Date',
        yaxis_title='YoY Change (%)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=500 if anomaly_detection and anomaly_detection.get('enabled') else 400,
        yaxis=dict(ticksuffix='%')
    )
    
    return fig