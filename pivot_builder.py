"""Interactive pivot table builder for FAO Food Price Index data."""

import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from io import BytesIO
import xlsxwriter


def prepare_temporal_dimensions(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add temporal dimension columns to the DataFrame.
    
    Args:
        df: Input DataFrame with date column
        date_col: Name of the date column
        
    Returns:
        DataFrame with additional Year, Quarter, Month columns
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['Year'] = df[date_col].dt.year
    df['Quarter'] = df[date_col].dt.quarter.map(lambda x: f'Q{x}')
    df['Month'] = df[date_col].dt.strftime('%Y-%m')
    
    return df


def validate_pivot_size(df: pd.DataFrame, row_dim: str, col_indices: List[str], max_cells: int = 1000) -> Tuple[bool, int, str]:
    """
    Validate that the pivot table won't exceed maximum cell count.
    
    Args:
        df: Input DataFrame
        row_dim: Row dimension column name
        col_indices: List of column indices to pivot
        max_cells: Maximum allowed cells
        
    Returns:
        Tuple of (is_valid, estimated_cells, warning_message)
    """
    if df.empty or not col_indices:
        return True, 0, ""
    
    # Count unique values in row dimension
    unique_rows = df[row_dim].nunique() if row_dim in df.columns else 1
    
    # Number of columns will be the number of selected indices
    unique_cols = len(col_indices)
    
    estimated_cells = unique_rows * unique_cols
    
    is_valid = estimated_cells <= max_cells
    
    if not is_valid:
        warning_msg = f"Estimated {estimated_cells} cells exceeds limit of {max_cells}. Please reduce selection."
    else:
        warning_msg = f"Estimated {estimated_cells} cells (within {max_cells} limit)"
    
    return is_valid, estimated_cells, warning_msg


def create_pivot_table(
    df: pd.DataFrame, 
    row_dim: str, 
    col_indices: List[str], 
    agg_func: str,
    index_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Create a pivot table from the FAO data.
    
    Args:
        df: Input DataFrame with FAO data
        row_dim: Row dimension ('Year', 'Quarter', 'Month')
        col_indices: List of price index columns to include
        agg_func: Aggregation function ('mean', 'max', 'min')
        index_mapping: Mapping from display names to column names
        
    Returns:
        Pivot table DataFrame
    """
    if df.empty or not col_indices:
        return pd.DataFrame()
    
    # Prepare data with temporal dimensions
    df_prep = prepare_temporal_dimensions(df)
    
    # Convert display names to column names
    column_names = [index_mapping[idx] for idx in col_indices if idx in index_mapping]
    
    # Filter to only include existing columns
    available_columns = [col for col in column_names if col in df_prep.columns]
    
    if not available_columns:
        return pd.DataFrame()
    
    # Create pivot table
    try:
        pivot_df = df_prep.pivot_table(
            index=row_dim,
            values=available_columns,
            aggfunc=agg_func,
            fill_value=0
        )
        
        # Reset index to make row dimension a regular column
        pivot_df = pivot_df.reset_index()
        
        # Round numeric values to 1 decimal place
        numeric_columns = pivot_df.select_dtypes(include=['float64', 'int64']).columns
        pivot_df[numeric_columns] = pivot_df[numeric_columns].round(1)
        
        # Rename columns back to display names for better UX
        reverse_mapping = {col: name for name, col in index_mapping.items() if col in available_columns}
        column_rename = {}
        for col in pivot_df.columns:
            if col in reverse_mapping:
                column_rename[col] = reverse_mapping[col]
        
        if column_rename:
            pivot_df = pivot_df.rename(columns=column_rename)
        
        return pivot_df
        
    except Exception as e:
        st.error(f"Error creating pivot table: {str(e)}")
        return pd.DataFrame()


def configure_aggrid_options(df: pd.DataFrame, row_dim: str) -> Dict[str, Any]:
    """
    Configure AgGrid options for the pivot table display.
    
    Args:
        df: Pivot table DataFrame
        row_dim: Row dimension name
        
    Returns:
        Dictionary of AgGrid options
    """
    if df.empty:
        return {}
    
    gb = GridOptionsBuilder.from_dataframe(df)
    
    # Configure the row dimension column
    gb.configure_column(row_dim, 
                       pinned='left', 
                       width=120,
                       headerName=row_dim,
                       cellStyle=JsCode("""
                       function(params) {
                           return {'background-color': '#f0f2f6', 'font-weight': 'bold'};
                       }
                       """))
    
    # Configure numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != row_dim:
            gb.configure_column(col,
                              type=['numericColumn'],
                              valueFormatter=JsCode("function(params) { return params.value != null ? params.value.toFixed(1) : ''; }"),
                              width=100,
                              cellStyle=JsCode("""
                              function(params) {
                                  if (params.value > 120) {
                                      return {'background-color': '#ffebee'};
                                  } else if (params.value < 80) {
                                      return {'background-color': '#e8f5e8'};
                                  }
                                  return null;
                              }
                              """))
    
    # Grid options
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_default_column(resizable=True, sortable=True, filterable=True)
    gb.configure_grid_options(enableRangeSelection=True)
    
    return gb.build()


def export_pivot_to_excel(
    pivot_df: pd.DataFrame,
    row_dimension: str,
    selected_indices: List[str],
    aggregation: str,
    original_df: pd.DataFrame,
    index_mapping: Dict[str, str]
) -> BytesIO:
    """
    Export pivot table to professionally formatted Excel workbook.
    
    Args:
        pivot_df: The pivot table DataFrame
        row_dimension: Row dimension used ('Year', 'Quarter', 'Month')
        selected_indices: List of selected price indices
        aggregation: Aggregation function used ('mean', 'max', 'min')
        original_df: Original source DataFrame
        index_mapping: Mapping from display names to column names
        
    Returns:
        BytesIO: Excel file content ready for download
    """
    output = BytesIO()
    
    # Create workbook with formatting options
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    
    # Define comprehensive formatting styles
    formats = {
        'title': workbook.add_format({
            'bold': True,
            'font_size': 16,
            'font_color': '#1f4e79',
            'align': 'center'
        }),
        'header': workbook.add_format({
            'bold': True,
            'font_color': '#FFFFFF',
            'bg_color': '#4472C4',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        }),
        'subheader': workbook.add_format({
            'bold': True,
            'font_color': '#FFFFFF',
            'bg_color': '#70AD47',
            'border': 1,
            'align': 'center'
        }),
        'number': workbook.add_format({
            'num_format': '#,##0.0',
            'border': 1,
            'align': 'right'
        }),
        'dimension': workbook.add_format({
            'bold': True,
            'bg_color': '#F2F2F2',
            'border': 1,
            'align': 'center'
        }),
        'metadata': workbook.add_format({
            'font_color': '#666666',
            'italic': True
        })
    }
    
    # Sheet 1: Main Pivot Table
    worksheet_pivot = workbook.add_worksheet('Pivot_Analysis')
    
    # Title and metadata
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    worksheet_pivot.write('A1', f'FAO Food Price Index - Pivot Analysis', formats['title'])
    worksheet_pivot.write('A2', f'Generated: {current_time}', formats['metadata'])
    worksheet_pivot.write('A3', f'Dimension: {row_dimension} | Aggregation: {aggregation.title()}', formats['metadata'])
    
    # Pivot table starting from row 5
    start_row = 4
    
    # Write headers
    for col_idx, column_name in enumerate(pivot_df.columns):
        if col_idx == 0:  # Row dimension column
            worksheet_pivot.write(start_row, col_idx, column_name, formats['dimension'])
        else:  # Value columns
            worksheet_pivot.write(start_row, col_idx, column_name, formats['header'])
    
    # Write data with conditional formatting
    for row_idx, (_, row) in enumerate(pivot_df.iterrows(), start=start_row + 1):
        for col_idx, value in enumerate(row):
            if col_idx == 0:  # Row dimension
                worksheet_pivot.write(row_idx, col_idx, value, formats['dimension'])
            else:  # Numeric values
                if pd.isna(value):
                    worksheet_pivot.write(row_idx, col_idx, "", formats['number'])
                else:
                    worksheet_pivot.write(row_idx, col_idx, float(value), formats['number'])
    
    # Add conditional formatting for value ranges
    if len(pivot_df) > 0:
        numeric_cols = pivot_df.select_dtypes(include=['float64', 'int64']).columns
        for col_idx, col_name in enumerate(pivot_df.columns):
            if col_name in numeric_cols:
                # Apply color scale from red (low) to green (high)
                worksheet_pivot.conditional_format(
                    start_row + 1, col_idx, 
                    start_row + len(pivot_df), col_idx,
                    {
                        'type': '3_color_scale',
                        'min_color': '#F8696B',
                        'mid_color': '#FFEB84', 
                        'max_color': '#63BE7B'
                    }
                )
    
    # Auto-adjust column widths
    for col_idx, column_name in enumerate(pivot_df.columns):
        if col_idx == 0:
            worksheet_pivot.set_column(col_idx, col_idx, 15)  # Row dimension
        else:
            worksheet_pivot.set_column(col_idx, col_idx, 12)  # Value columns
    
    # Freeze panes
    worksheet_pivot.freeze_panes(start_row + 1, 1)
    
    # Sheet 2: Configuration Details
    worksheet_config = workbook.add_worksheet('Configuration')
    
    config_data = [
        ['Configuration Item', 'Value'],
        ['Export Date', current_time],
        ['Row Dimension', row_dimension],
        ['Aggregation Function', aggregation.title()],
        ['Selected Indices', ', '.join(selected_indices)],
        ['Total Data Points', len(pivot_df)],
        ['Columns in Pivot', len(pivot_df.columns)],
        ['Source Data Range', f"{original_df['date'].min().strftime('%Y-%m')} to {original_df['date'].max().strftime('%Y-%m')}"],
        ['Source Records', len(original_df)]
    ]
    
    for row_idx, (item, value) in enumerate(config_data):
        if row_idx == 0:  # Header
            worksheet_config.write(row_idx, 0, item, formats['header'])
            worksheet_config.write(row_idx, 1, value, formats['header'])
        else:
            worksheet_config.write(row_idx, 0, item, formats['dimension'])
            worksheet_config.write(row_idx, 1, str(value))
    
    worksheet_config.set_column(0, 0, 20)
    worksheet_config.set_column(1, 1, 30)
    
    # Sheet 3: Statistics
    if len(pivot_df) > 1:
        worksheet_stats = workbook.add_worksheet('Statistics')
        
        numeric_cols = pivot_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            stats_df = pivot_df[numeric_cols].describe()
            
            # Write statistics header
            worksheet_stats.write('A1', 'Pivot Table Statistics', formats['title'])
            
            # Statistics table starting from row 3
            stats_start_row = 2
            
            # Write column headers
            worksheet_stats.write(stats_start_row, 0, 'Statistic', formats['header'])
            for col_idx, col_name in enumerate(numeric_cols, 1):
                worksheet_stats.write(stats_start_row, col_idx, col_name, formats['header'])
            
            # Write statistics data
            for row_idx, (stat_name, row) in enumerate(stats_df.iterrows(), stats_start_row + 1):
                worksheet_stats.write(row_idx, 0, stat_name.title(), formats['dimension'])
                for col_idx, value in enumerate(row, 1):
                    worksheet_stats.write(row_idx, col_idx, float(value), formats['number'])
            
            # Format columns
            worksheet_stats.set_column(0, 0, 15)
            for col_idx in range(1, len(numeric_cols) + 1):
                worksheet_stats.set_column(col_idx, col_idx, 12)
    
    # Sheet 4: Raw Pivot Data (for advanced users)
    worksheet_raw = workbook.add_worksheet('Raw_Data')
    
    # Write raw pivot data without formatting for data analysis
    for col_idx, column_name in enumerate(pivot_df.columns):
        worksheet_raw.write(0, col_idx, column_name, formats['header'])
    
    for row_idx, (_, row) in enumerate(pivot_df.iterrows(), 1):
        for col_idx, value in enumerate(row):
            if pd.isna(value):
                worksheet_raw.write(row_idx, col_idx, "")
            else:
                worksheet_raw.write(row_idx, col_idx, value)
    
    # Auto-fit columns in raw data
    for col_idx in range(len(pivot_df.columns)):
        worksheet_raw.set_column(col_idx, col_idx, 12)
    
    workbook.close()
    output.seek(0)
    
    return output


def render_pivot_interface(df: pd.DataFrame, index_mapping: Dict[str, str]) -> None:
    """
    Render the complete pivot table interface.
    
    Args:
        df: Input DataFrame with FAO data
        index_mapping: Mapping from display names to column names
    """
    if df is None or df.empty:
        st.warning("No data available for pivot analysis")
        return
    
    st.markdown("### 🔧 Pivot Configuration")
    
    # Create three columns for controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Row Dimension**")
        row_dimension = st.radio(
            "Select row grouping:",
            options=['Year', 'Quarter', 'Month'],
            index=0,
            key='pivot_row_dim',
            help="Choose how to group data by time period"
        )
    
    with col2:
        st.markdown("**Columns (Indices)**")
        available_indices = list(index_mapping.keys())
        selected_indices = st.multiselect(
            "Select price indices:",
            options=available_indices,
            default=['Food Price Index', 'Meat', 'Dairy'],
            key='pivot_col_indices',
            help="Choose which price indices to display as columns"
        )
    
    with col3:
        st.markdown("**Aggregation**")
        aggregation = st.radio(
            "Select aggregation:",
            options=['mean', 'max', 'min'],
            index=0,
            key='pivot_agg',
            help="Choose how to aggregate values for each cell"
        )
    
    # Validation section
    st.markdown("### 📊 Pivot Preview")
    
    if not selected_indices:
        st.warning("Please select at least one price index to create the pivot table.")
        return
    
    # Validate pivot size
    is_valid, cell_count, warning_msg = validate_pivot_size(df, row_dimension, selected_indices)
    
    # Display validation result
    if is_valid:
        st.success(f"✅ {warning_msg}")
    else:
        st.error(f"❌ {warning_msg}")
        st.info("💡 Try reducing the number of selected indices or using a different row dimension.")
        return
    
    # Create and display pivot table
    with st.spinner("Creating pivot table..."):
        pivot_df = create_pivot_table(df, row_dimension, selected_indices, aggregation, index_mapping)
    
    if pivot_df.empty:
        st.warning("Unable to create pivot table with current selection.")
        return
    
    # Display summary
    st.info(f"📈 Showing {len(pivot_df)} {row_dimension.lower()} periods × {len(selected_indices)} indices = {len(pivot_df) * len(selected_indices)} data points")
    
    # Configure and display AgGrid
    try:
        grid_options = configure_aggrid_options(pivot_df, row_dimension)
        
        st.markdown("### 📋 Interactive Pivot Table")
        
        # Display the AgGrid
        AgGrid(
            pivot_df,
            gridOptions=grid_options,
            height=400,
            theme='streamlit',
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            key='pivot_aggrid'
        )
        
        # Export options
        st.markdown("### 📥 Export Options")
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📊 Export Pivot to Excel", key='pivot_excel', type='primary'):
                try:
                    with st.spinner("📊 Creating Excel export..."):
                        # Generate Excel file with comprehensive formatting
                        excel_data = export_pivot_to_excel(
                            pivot_df=pivot_df,
                            row_dimension=row_dimension,
                            selected_indices=selected_indices,
                            aggregation=aggregation,
                            original_df=df,
                            index_mapping=index_mapping
                        )
                        
                        # Create filename with timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"FAO_Pivot_{row_dimension}_{aggregation}_{timestamp}.xlsx"
                        
                        # Provide download button
                        st.download_button(
                            label="💾 Download Excel Report",
                            data=excel_data.getvalue(),
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help=f"Professional Excel report with pivot analysis, configuration, and statistics"
                        )
                        
                        st.success("✅ Excel report generated successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Excel export failed: {str(e)}")
                    st.info("💡 Falling back to CSV export option...")
        
        with col_export2:
            if st.button("📄 Download as CSV", key='pivot_csv'):
                csv_data = pivot_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"fao_pivot_{row_dimension}_{aggregation}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col_export3:
            # Show basic statistics
            if st.button("📈 Show Statistics", key='pivot_stats'):
                st.markdown("**Pivot Table Statistics:**")
                numeric_cols = pivot_df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:  # Exclude row dimension
                    stats_df = pivot_df[numeric_cols].describe().round(2)
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No numeric data available for statistics.")
        
        # Export information
        st.markdown("**📋 Export Information:**")
        export_info_cols = st.columns(3)
        with export_info_cols[0]:
            st.caption(f"📊 **Pivot:** {len(pivot_df)} rows × {len(selected_indices)} indices")
        with export_info_cols[1]:
            st.caption(f"🔧 **Aggregation:** {aggregation.title()}")
        with export_info_cols[2]:
            st.caption(f"📅 **Dimension:** {row_dimension}")
    
    except Exception as e:
        st.error(f"Error displaying pivot table: {str(e)}")
        st.info("Falling back to simple table display...")
        
        # Fallback to simple dataframe display
        st.dataframe(pivot_df, use_container_width=True)


# Example usage and testing function
def test_pivot_builder():
    """Test function for development purposes."""
    # Create sample data
    sample_data = {
        'date': pd.date_range('2020-01-01', '2023-12-01', freq='M'),
        'food_price_index': range(100, 148),
        'meat': range(95, 143),
        'dairy': range(105, 153),
        'cereals': range(90, 138),
        'oils': range(110, 158),
        'sugar': range(85, 133)
    }
    
    df = pd.DataFrame(sample_data)
    
    index_mapping = {
        'Food Price Index': 'food_price_index',
        'Meat': 'meat',
        'Dairy': 'dairy',
        'Cereals': 'cereals',
        'Oils': 'oils',
        'Sugar': 'sugar'
    }
    
    return df, index_mapping


if __name__ == "__main__":
    # For testing purposes
    st.title("Pivot Builder Test")
    df, mapping = test_pivot_builder()
    render_pivot_interface(df, mapping)