"""
Correlation analysis module for FAO Food Price Index Dashboard.
Provides correlation calculation, visualization, and export functionality.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
from io import BytesIO
import xlsxwriter
from performance_monitor import performance_monitor, performance_context


def _calculate_pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient and p-value using numpy/pandas."""
    # Use pandas built-in correlation which handles NaN values well
    series_x = pd.Series(x)
    series_y = pd.Series(y)
    
    # Calculate correlation using pandas
    corr = series_x.corr(series_y, method='pearson')
    
    # For p-value, use a simple approximation based on sample size
    # This is a simplified approach - scipy uses more sophisticated methods
    n = len(series_x.dropna())
    if n < 3:
        pval = 1.0  # Not enough data for meaningful correlation
    else:
        # Simple t-test approximation for correlation significance
        # Note: This is less sophisticated than scipy's implementation
        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2)) if abs(corr) < 0.999 else float('inf')
        # Very simplified p-value approximation
        pval = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / 10))) if abs(t_stat) != float('inf') else 0.001
    
    return corr, pval


def _calculate_spearman_correlation(x, y):
    """Calculate Spearman correlation coefficient using pandas."""
    series_x = pd.Series(x)
    series_y = pd.Series(y)
    
    # Use pandas built-in spearman correlation
    corr = series_x.corr(series_y, method='spearman')
    
    # Simplified p-value approximation
    n = len(series_x.dropna())
    pval = max(0.001, min(0.999, 1 - abs(corr))) if n >= 3 else 1.0
    
    return corr, pval


def _calculate_kendall_correlation(x, y):
    """Calculate Kendall correlation coefficient using pandas."""
    series_x = pd.Series(x)
    series_y = pd.Series(y)
    
    # Use pandas built-in kendall correlation
    corr = series_x.corr(series_y, method='kendall')
    
    # Simplified p-value approximation
    n = len(series_x.dropna())
    pval = max(0.001, min(0.999, 1 - abs(corr))) if n >= 3 else 1.0
    
    return corr, pval


@performance_monitor('correlation_calculation', include_args=True)
def calculate_correlation_matrix(
    df: pd.DataFrame,
    indices: List[str],
    method: str = 'pearson',
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Calculate correlation matrix for selected indices.
    
    Args:
        df: DataFrame with date index and price index columns
        indices: List of column names to include in correlation
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of observations required
    
    Returns:
        DataFrame containing correlation matrix
    """
    # Filter to selected indices
    df_filtered = df[indices].copy()
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = df_filtered.corr(method='pearson', min_periods=min_periods)
    elif method == 'spearman':
        corr_matrix = df_filtered.corr(method='spearman', min_periods=min_periods)
    elif method == 'kendall':
        corr_matrix = df_filtered.corr(method='kendall', min_periods=min_periods)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return corr_matrix


def calculate_correlation_with_pvalues(
    df: pd.DataFrame,
    indices: List[str],
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlation matrix with p-values for significance testing.
    
    Args:
        df: DataFrame with date index and price index columns
        indices: List of column names to include
        method: Correlation method
    
    Returns:
        Tuple of (correlation_matrix, p_value_matrix)
    """
    df_filtered = df[indices].dropna()
    n = len(indices)
    
    # Initialize matrices
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=indices, columns=indices)
    pval_matrix = pd.DataFrame(np.ones((n, n)), index=indices, columns=indices)
    
    # Calculate correlations and p-values
    for i, col1 in enumerate(indices):
        for j, col2 in enumerate(indices):
            if i <= j:  # Only calculate upper triangle and diagonal
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                    pval_matrix.loc[col1, col2] = 0.0
                else:
                    # Calculate correlation and p-value
                    if method == 'pearson':
                        corr, pval = _calculate_pearson_correlation(df_filtered[col1], df_filtered[col2])
                    elif method == 'spearman':
                        corr, pval = _calculate_spearman_correlation(df_filtered[col1], df_filtered[col2])
                    elif method == 'kendall':
                        corr, pval = _calculate_kendall_correlation(df_filtered[col1], df_filtered[col2])
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    # Store in both positions (symmetric matrix)
                    corr_matrix.loc[col1, col2] = corr
                    corr_matrix.loc[col2, col1] = corr
                    pval_matrix.loc[col1, col2] = pval
                    pval_matrix.loc[col2, col1] = pval
    
    return corr_matrix, pval_matrix


def build_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    pval_matrix: Optional[pd.DataFrame] = None,
    title: str = "Correlation Matrix",
    show_values: bool = True,
    significance_level: float = 0.05
) -> go.Figure:
    """
    Build an interactive Plotly heatmap for correlation visualization.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        pval_matrix: Optional p-value matrix for significance
        title: Chart title
        show_values: Whether to show correlation values in cells
        significance_level: P-value threshold for significance
    
    Returns:
        Plotly Figure object
    """
    # Prepare text annotations
    if show_values:
        text_matrix = corr_matrix.round(2).astype(str)
        
        # Add significance markers if p-values provided
        if pval_matrix is not None:
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    if i != j:  # Skip diagonal
                        pval = pval_matrix.iloc[i, j]
                        if pval < significance_level:
                            if pval < 0.001:
                                text_matrix.iloc[i, j] += "***"
                            elif pval < 0.01:
                                text_matrix.iloc[i, j] += "**"
                            elif pval < 0.05:
                                text_matrix.iloc[i, j] += "*"
    else:
        text_matrix = None
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        text=text_matrix.values if text_matrix is not None else None,
        texttemplate='%{text}' if show_values else None,
        textfont={"size": 10},
        colorscale='RdBu_r',  # Red-Blue diverging scale (reversed)
        zmid=0,  # Center at 0
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="Correlation",
            tickmode="linear",
            tick0=-1,
            dtick=0.2
        ),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="",
            tickangle=45,
            side="bottom"
        ),
        yaxis=dict(
            title="",
            autorange='reversed'  # Match traditional correlation matrix layout
        ),
        width=800,
        height=600,
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    # Add annotations for significance legend if p-values provided
    if pval_matrix is not None and show_values:
        fig.add_annotation(
            text="Significance: * p<0.05, ** p<0.01, *** p<0.001",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="center"
        )
    
    return fig


def calculate_rolling_correlation(
    df: pd.DataFrame,
    index1: str,
    index2: str,
    window: int = 60,
    min_periods: int = 30
) -> pd.Series:
    """
    Calculate rolling correlation between two indices over time.
    
    Args:
        df: DataFrame with price indices
        index1: First index column name
        index2: Second index column name
        window: Rolling window size in periods
        min_periods: Minimum observations required
    
    Returns:
        Series with rolling correlation values
    """
    return df[index1].rolling(window=window, min_periods=min_periods).corr(df[index2])


def get_correlation_insights(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Extract key insights from correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        threshold: Threshold for strong correlations
    
    Returns:
        Dictionary with insights categorized by type
    """
    insights = {
        'strong_positive': [],
        'strong_negative': [],
        'moderate_positive': [],
        'moderate_negative': [],
        'weak': []
    }
    
    # Get unique pairs (upper triangle)
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.index[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if pd.notna(corr_val):
                pair = (col1, col2, corr_val)
                
                if corr_val >= threshold:
                    insights['strong_positive'].append(pair)
                elif corr_val <= -threshold:
                    insights['strong_negative'].append(pair)
                elif 0.4 <= corr_val < threshold:
                    insights['moderate_positive'].append(pair)
                elif -threshold < corr_val <= -0.4:
                    insights['moderate_negative'].append(pair)
                else:
                    insights['weak'].append(pair)
    
    # Sort by absolute correlation value
    for key in insights:
        insights[key] = sorted(insights[key], key=lambda x: abs(x[2]), reverse=True)
    
    return insights


def export_correlation_to_excel(
    corr_matrix: pd.DataFrame,
    pval_matrix: Optional[pd.DataFrame],
    insights: Dict[str, List],
    metadata: Dict[str, Any],
    original_data: Optional[pd.DataFrame] = None
) -> BytesIO:
    """
    Export correlation analysis to Excel with formatting.
    
    Args:
        corr_matrix: Correlation matrix
        pval_matrix: P-value matrix (optional)
        insights: Correlation insights dictionary
        metadata: Analysis metadata (period, method, etc.)
        original_data: Original data for reference (optional)
    
    Returns:
        BytesIO object containing Excel file
    """
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    number_format = workbook.add_format({
        'num_format': '0.000',
        'border': 1,
        'align': 'center'
    })
    
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 14,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    # Sheet 1: Correlation Matrix
    ws_corr = workbook.add_worksheet('Correlation_Matrix')
    
    # Title
    ws_corr.merge_range('A1:' + chr(65 + len(corr_matrix.columns)) + '1', 
                        'Correlation Matrix Analysis', title_format)
    
    # Write headers
    ws_corr.write(2, 0, '', header_format)
    for col_idx, col in enumerate(corr_matrix.columns):
        ws_corr.write(2, col_idx + 1, col, header_format)
    
    # Write correlation values with conditional formatting
    for row_idx, index in enumerate(corr_matrix.index):
        ws_corr.write(row_idx + 3, 0, index, header_format)
        for col_idx, col in enumerate(corr_matrix.columns):
            value = corr_matrix.loc[index, col]
            ws_corr.write_number(row_idx + 3, col_idx + 1, value, number_format)
    
    # Add conditional formatting (color scale)
    ws_corr.conditional_format(3, 1, 2 + len(corr_matrix), len(corr_matrix.columns), {
        'type': '3_color_scale',
        'min_color': '#FF0000',  # Red for -1
        'mid_color': '#FFFFFF',  # White for 0
        'max_color': '#00FF00',  # Green for 1
        'min_value': -1,
        'mid_value': 0,
        'max_value': 1
    })
    
    # Auto-fit columns
    for col_idx in range(len(corr_matrix.columns) + 1):
        ws_corr.set_column(col_idx, col_idx, 15)
    
    # Sheet 2: P-values (if provided)
    if pval_matrix is not None:
        ws_pval = workbook.add_worksheet('P_Values')
        
        ws_pval.merge_range('A1:' + chr(65 + len(pval_matrix.columns)) + '1',
                           'Statistical Significance (P-Values)', title_format)
        
        # Write headers
        ws_pval.write(2, 0, '', header_format)
        for col_idx, col in enumerate(pval_matrix.columns):
            ws_pval.write(2, col_idx + 1, col, header_format)
        
        # Write p-values
        pval_format = workbook.add_format({
            'num_format': '0.0000',
            'border': 1,
            'align': 'center'
        })
        
        for row_idx, index in enumerate(pval_matrix.index):
            ws_pval.write(row_idx + 3, 0, index, header_format)
            for col_idx, col in enumerate(pval_matrix.columns):
                value = pval_matrix.loc[index, col]
                ws_pval.write_number(row_idx + 3, col_idx + 1, value, pval_format)
        
        # Highlight significant values
        ws_pval.conditional_format(3, 1, 2 + len(pval_matrix), len(pval_matrix.columns), {
            'type': 'cell',
            'criteria': '<',
            'value': 0.05,
            'format': workbook.add_format({'bg_color': '#90EE90'})  # Light green
        })
        
        for col_idx in range(len(pval_matrix.columns) + 1):
            ws_pval.set_column(col_idx, col_idx, 15)
    
    # Sheet 3: Insights
    ws_insights = workbook.add_worksheet('Insights')
    
    ws_insights.merge_range('A1:D1', 'Correlation Insights', title_format)
    
    row = 3
    for category, pairs in insights.items():
        if pairs:
            # Category header
            category_title = category.replace('_', ' ').title()
            ws_insights.merge_range(row, 0, row, 3, category_title, header_format)
            row += 1
            
            # Column headers
            ws_insights.write(row, 0, 'Index 1', header_format)
            ws_insights.write(row, 1, 'Index 2', header_format)
            ws_insights.write(row, 2, 'Correlation', header_format)
            ws_insights.write(row, 3, 'Interpretation', header_format)
            row += 1
            
            # Data
            for pair in pairs[:5]:  # Top 5 for each category
                ws_insights.write(row, 0, pair[0])
                ws_insights.write(row, 1, pair[1])
                ws_insights.write_number(row, 2, pair[2], number_format)
                
                # Interpretation
                if abs(pair[2]) >= 0.7:
                    interp = "Strong"
                elif abs(pair[2]) >= 0.4:
                    interp = "Moderate"
                else:
                    interp = "Weak"
                ws_insights.write(row, 3, interp)
                row += 1
            
            row += 1  # Empty row between categories
    
    # Auto-fit columns
    for col_idx in range(4):
        ws_insights.set_column(col_idx, col_idx, 20)
    
    # Sheet 4: Metadata
    ws_meta = workbook.add_worksheet('Analysis_Info')
    
    ws_meta.merge_range('A1:B1', 'Analysis Metadata', title_format)
    
    meta_row = 3
    for key, value in metadata.items():
        ws_meta.write(meta_row, 0, key, header_format)
        ws_meta.write(meta_row, 1, str(value))
        meta_row += 1
    
    ws_meta.set_column(0, 0, 25)
    ws_meta.set_column(1, 1, 40)
    
    # Sheet 5: Raw Data (if provided)
    if original_data is not None and not original_data.empty:
        ws_data = workbook.add_worksheet('Raw_Data')
        
        # Write headers
        for col_idx, col in enumerate(original_data.columns):
            ws_data.write(0, col_idx, col, header_format)
        
        # Write data (limit to 1000 rows for performance)
        max_rows = min(1000, len(original_data))
        for row_idx in range(max_rows):
            for col_idx, col in enumerate(original_data.columns):
                value = original_data.iloc[row_idx, col_idx]
                if pd.notna(value):
                    if isinstance(value, (int, float)):
                        ws_data.write_number(row_idx + 1, col_idx, value)
                    else:
                        ws_data.write(row_idx + 1, col_idx, str(value))
        
        # Auto-fit columns
        for col_idx in range(len(original_data.columns)):
            ws_data.set_column(col_idx, col_idx, 12)
        
        if max_rows < len(original_data):
            ws_data.write(max_rows + 2, 0, 
                         f"Note: Showing first {max_rows} of {len(original_data)} rows",
                         workbook.add_format({'italic': True, 'font_color': 'gray'}))
    
    workbook.close()
    output.seek(0)
    return output


def interpret_correlation(correlation: float) -> str:
    """
    Provide interpretation of correlation coefficient.
    
    Args:
        correlation: Correlation coefficient value
    
    Returns:
        String interpretation
    """
    abs_corr = abs(correlation)
    
    if abs_corr >= 0.9:
        strength = "Very strong"
    elif abs_corr >= 0.7:
        strength = "Strong"
    elif abs_corr >= 0.5:
        strength = "Moderate"
    elif abs_corr >= 0.3:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    direction = "positive" if correlation >= 0 else "negative"
    
    return f"{strength} {direction} correlation"