"""FAO Food Price Index Dashboard using Streamlit."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from data_pipeline import DataPipeline
from chart_builder import build_chart
from kpi_calculator import calculate_kpis, format_kpi_for_display, get_trend_emoji, get_delta_color, calculate_detailed_statistics
from excel_exporter import ExcelExporter
from pivot_builder import render_pivot_interface
from correlation_analyzer import (
    calculate_correlation_matrix,
    calculate_correlation_with_pvalues,
    build_correlation_heatmap,
    get_correlation_insights,
    export_correlation_to_excel,
    interpret_correlation
)
from anomaly_detector import (
    get_anomaly_summary,
    get_anomaly_insights,
    classify_anomaly_severity
)
from performance_monitor import performance_monitor, performance_context


@st.cache_data(ttl=3600)  # Cache for 1 hour
@performance_monitor('streamlit_data_loading', include_args=False)
def load_fao_data():
    """
    Load FAO data using DataPipeline with caching and error handling.
    
    Returns:
        tuple: (DataFrame or None, error_message or None)
    """
    try:
        with st.spinner("📊 Loading FAO Food Price Index data..."):
            pipeline = DataPipeline(sheet_name='Monthly', cache_ttl_hours=1.0)
            df = pipeline.run()
            
            if df is not None and len(df) > 0:
                return df, None
            else:
                return None, "No data available from FAO source"
                
    except Exception as e:
        error_msg = str(e)
        # Try to provide user-friendly error messages
        if "Excel validation failed" in error_msg:
            error_msg = "Unable to fetch fresh data from FAO. This may be due to server issues or connectivity problems."
        elif "Network error" in error_msg or "ConnectionError" in error_msg:
            error_msg = "Network connection issue. Please check your internet connection and try again."
        
        return None, error_msg


def render_correlation_analysis(df: pd.DataFrame, index_mapping: Dict, selected_indices: List[str]):
    """
    Render the correlation analysis tab.
    
    Args:
        df: DataFrame with FAO data
        index_mapping: Mapping of display names to column names
        selected_indices: Currently selected indices from sidebar
    """
    st.header("🔗 Correlation Analysis")
    
    if df is None or df.empty:
        st.warning("📊 Please load data to perform correlation analysis")
        return
    
    st.markdown("""
    Analyze correlations between different food price indices to understand their relationships and dependencies.
    This analysis helps identify which commodities move together and can inform risk management strategies.
    """)
    
    # Settings section
    st.markdown("### ⚙️ Analysis Settings")
    
    col_settings1, col_settings2, col_settings3 = st.columns(3)
    
    with col_settings1:
        st.markdown("**📅 Time Period**")
        
        # Date range selector
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        # Presets
        preset = st.selectbox(
            "Quick select",
            ["Custom", "Last 1 Year", "Last 3 Years", "Last 5 Years", "All Time"],
            key="corr_preset"
        )
        
        if preset == "Last 1 Year":
            start_date = max(min_date, max_date - pd.DateOffset(years=1))
            end_date = max_date
        elif preset == "Last 3 Years":
            start_date = max(min_date, max_date - pd.DateOffset(years=3))
            end_date = max_date
        elif preset == "Last 5 Years":
            start_date = max(min_date, max_date - pd.DateOffset(years=5))
            end_date = max_date
        elif preset == "All Time":
            start_date = min_date
            end_date = max_date
        else:  # Custom
            date_range = st.date_input(
                "Select range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="corr_date_range"
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
    
    with col_settings2:
        st.markdown("**📊 Index Selection**")
        
        # Multi-select for correlation indices
        corr_indices = st.multiselect(
            "Select indices (min 2)",
            options=list(index_mapping.keys()),
            default=selected_indices[:5] if len(selected_indices) >= 2 else list(index_mapping.keys())[:5],
            key="corr_indices",
            help="Select at least 2 indices to calculate correlations"
        )
        
        # Select all/clear buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Select All", key="corr_select_all"):
                st.session_state.corr_indices = list(index_mapping.keys())
                st.rerun()
        with col_btn2:
            if st.button("Clear", key="corr_clear"):
                st.session_state.corr_indices = []
                st.rerun()
    
    with col_settings3:
        st.markdown("**🔧 Analysis Options**")
        
        # Correlation method
        method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
            index=0,
            key="corr_method",
            help="Pearson: Linear correlation, Spearman: Rank correlation, Kendall: Ordinal association"
        )
        
        # Show significance
        show_significance = st.checkbox(
            "Show significance levels",
            value=True,
            key="corr_significance",
            help="Display statistical significance indicators"
        )
        
        # Show values in heatmap
        show_values = st.checkbox(
            "Show values in heatmap",
            value=True,
            key="corr_show_values"
        )
    
    # Validate selections
    if len(corr_indices) < 2:
        st.warning("⚠️ Please select at least 2 indices to calculate correlations")
        return
    
    # Filter data
    df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)].copy()
    
    if df_filtered.empty:
        st.error("❌ No data available for the selected time period")
        return
    
    # Get column names for selected indices
    selected_columns = [index_mapping[idx] for idx in corr_indices if idx in index_mapping]
    
    # Ensure columns exist in dataframe
    available_columns = [col for col in selected_columns if col in df_filtered.columns]
    
    if len(available_columns) < 2:
        st.error("❌ Not enough data columns available for correlation analysis")
        return
    
    st.markdown("---")
    
    # Calculate correlations
    try:
        with st.spinner("🔄 Calculating correlations..."):
            if show_significance:
                corr_matrix, pval_matrix = calculate_correlation_with_pvalues(
                    df_filtered, available_columns, method
                )
            else:
                corr_matrix = calculate_correlation_matrix(
                    df_filtered, available_columns, method
                )
                pval_matrix = None
            
            # Rename index and columns for display
            display_names = {col: name for name, col in index_mapping.items() if col in available_columns}
            corr_matrix.index = [display_names.get(idx, idx) for idx in corr_matrix.index]
            corr_matrix.columns = [display_names.get(col, col) for col in corr_matrix.columns]
            
            if pval_matrix is not None:
                pval_matrix.index = corr_matrix.index
                pval_matrix.columns = corr_matrix.columns
        
        # Display correlation heatmap
        st.markdown("### 📊 Correlation Heatmap")
        
        fig = build_correlation_heatmap(
            corr_matrix,
            pval_matrix if show_significance else None,
            title=f"Correlation Matrix ({method.capitalize()}) - {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}",
            show_values=show_values,
            significance_level=0.05
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Get insights
        insights = get_correlation_insights(corr_matrix, threshold=0.7)
        
        # Display insights
        st.markdown("### 🔍 Key Insights")
        
        col_insights1, col_insights2 = st.columns(2)
        
        with col_insights1:
            st.markdown("**Strong Positive Correlations**")
            if insights['strong_positive']:
                for idx1, idx2, corr in insights['strong_positive'][:5]:
                    st.write(f"• {idx1} ↔ {idx2}: {corr:.3f}")
                    st.caption(interpret_correlation(corr))
            else:
                st.info("No strong positive correlations found")
            
            st.markdown("**Moderate Positive Correlations**")
            if insights['moderate_positive']:
                for idx1, idx2, corr in insights['moderate_positive'][:3]:
                    st.write(f"• {idx1} ↔ {idx2}: {corr:.3f}")
            else:
                st.info("No moderate positive correlations found")
        
        with col_insights2:
            st.markdown("**Strong Negative Correlations**")
            if insights['strong_negative']:
                for idx1, idx2, corr in insights['strong_negative'][:5]:
                    st.write(f"• {idx1} ↔ {idx2}: {corr:.3f}")
                    st.caption(interpret_correlation(corr))
            else:
                st.info("No strong negative correlations found")
            
            st.markdown("**Moderate Negative Correlations**")
            if insights['moderate_negative']:
                for idx1, idx2, corr in insights['moderate_negative'][:3]:
                    st.write(f"• {idx1} ↔ {idx2}: {corr:.3f}")
            else:
                st.info("No moderate negative correlations found")
        
        # Statistics summary
        st.markdown("### 📈 Statistical Summary")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Data Points", len(df_filtered))
            st.metric("Indices Analyzed", len(available_columns))
        
        with col_stats2:
            # Calculate average absolute correlation (excluding diagonal)
            mask = ~np.eye(len(corr_matrix), dtype=bool)
            avg_corr = np.abs(corr_matrix.values[mask]).mean()
            st.metric("Avg |Correlation|", f"{avg_corr:.3f}")
            
            # Count significant correlations if p-values available
            if pval_matrix is not None:
                sig_count = (pval_matrix.values[mask] < 0.05).sum()
                st.metric("Significant Pairs", f"{sig_count}/{len(corr_matrix) * (len(corr_matrix) - 1) // 2}")
        
        with col_stats3:
            max_corr = corr_matrix.values[mask].max()
            min_corr = corr_matrix.values[mask].min()
            st.metric("Max Correlation", f"{max_corr:.3f}")
            st.metric("Min Correlation", f"{min_corr:.3f}")
        
        # Export section
        st.markdown("---")
        st.markdown("### 📥 Export Options")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Excel export
            if st.button("📊 Export to Excel", key="corr_export_excel", type="primary"):
                try:
                    with st.spinner("Creating Excel report..."):
                        # Prepare metadata
                        metadata = {
                            "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Time Period": f"{start_date} to {end_date}",
                            "Method": method.capitalize(),
                            "Number of Indices": len(available_columns),
                            "Data Points": len(df_filtered),
                            "Indices": ", ".join(corr_matrix.index)
                        }
                        
                        # Prepare data for export
                        export_data = df_filtered[['date'] + available_columns].copy()
                        export_data.rename(columns=display_names, inplace=True)
                        
                        # Create Excel file
                        excel_data = export_correlation_to_excel(
                            corr_matrix,
                            pval_matrix,
                            insights,
                            metadata,
                            export_data
                        )
                        
                        # Download button
                        st.download_button(
                            label="💾 Download Excel Report",
                            data=excel_data.getvalue(),
                            file_name=f"Correlation_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("✅ Excel report generated successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error generating Excel report: {str(e)}")
        
        with col_export2:
            # CSV export
            if st.button("📄 Export to CSV", key="corr_export_csv"):
                csv_data = corr_matrix.to_csv()
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"Correlation_Matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Display raw correlation matrix
        with st.expander("📋 View Correlation Matrix", expanded=False):
            st.dataframe(
                corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1, axis=None)
                .format("{:.3f}"),
                use_container_width=True
            )
            
            if pval_matrix is not None:
                st.markdown("**P-Values Matrix**")
                st.dataframe(
                    pval_matrix.style.format("{:.4f}")
                    .applymap(lambda x: 'background-color: lightgreen' if x < 0.05 else ''),
                    use_container_width=True
                )
        
    except Exception as e:
        st.error(f"❌ Error calculating correlations: {str(e)}")
        st.info("Please check your data selection and try again.")


def main():
    """Main application function with error boundary."""
    try:
        # Page configuration
        st.set_page_config(
            page_title="FAO Food Price Index Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("📊 FAO Food Price Index Dashboard")
        
        # Load data
        df, error = load_fao_data()
        
        # Show data status
        if error:
            st.warning(f"⚠️ Data Loading Issue: {error}")
            if "server issues" in error.lower():
                st.info("💡 The system may fall back to cached data if available. Data processing pipeline is working correctly.")
        
        st.markdown("---")
        
        # Define index mapping (display name to column name) - moved outside sidebar
        index_mapping = {
            'Food Price Index': 'food_price_index',
            'Meat': 'meat', 
            'Dairy': 'dairy',
            'Cereals': 'cereals',
            'Oils': 'oils',
            'Sugar': 'sugar'
        }
        
        # Initialize selected_indices with default
        selected_indices = ['Food Price Index']
        
        # Sidebar
        with st.sidebar:
            st.header("📋 Dashboard Controls")
            
            # Date Range Filter
            st.markdown("### 📅 Date Range Filter")
            
            if df is not None and len(df) > 0:
                # Get min and max dates from data
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                
                # Calculate default start date (5 years before max date)
                from dateutil.relativedelta import relativedelta
                default_start_date = max_date - relativedelta(years=5)
                # Ensure default start date is not before min_date
                if default_start_date < min_date:
                    default_start_date = min_date
                
                # Date range selector with default last 5 years
                date_range = st.date_input(
                    "Select date range",
                    value=(default_start_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range",
                    help="Filter data by selecting start and end dates (Default: last 5 years)"
                )
                
                # Ensure we have both start and end dates
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    # Store original count for comparison
                    original_count = len(df)
                    
                    # Filter the dataframe
                    df = df[(df['date'].dt.date >= start_date) & 
                           (df['date'].dt.date <= end_date)].copy()
                    
                    # Show record count after filtering
                    filtered_count = len(df)
                    st.success(f"📊 Showing {filtered_count} records")
                    st.caption(f"Date range: {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}")
                    if original_count != filtered_count:
                        st.caption(f"Filtered from {original_count} total records")
                elif len(date_range) == 1:
                    st.warning("Please select both start and end dates")
            else:
                st.warning("No data available for filtering")
            
            st.markdown("---")
            
            # Index Selection
            st.markdown("### 📈 Index Selection")
            
            # Multi-select for indices with default (index_mapping defined above)
            selected_indices = st.multiselect(
                "Select indices to display",
                options=list(index_mapping.keys()),
                default=['Food Price Index'],
                key="index_selection",
                help="Choose which price indices to display on the chart"
            )
            
            # Select All / Clear buttons
            col_all, col_clear = st.columns(2)
            with col_all:
                if st.button("Select All", key="select_all"):
                    st.session_state.index_selection = list(index_mapping.keys())
                    st.rerun()
            
            with col_clear:
                if st.button("Clear", key="clear_selection"):
                    # Always keep at least Food Price Index selected
                    st.session_state.index_selection = ['Food Price Index']
                    st.rerun()
            
            # Validation: ensure at least one index is selected
            if not selected_indices:
                selected_indices = ['Food Price Index']
                st.warning("At least one index must be selected. Defaulting to Food Price Index.")
            
            # Show selected count
            st.caption(f"📊 {len(selected_indices)} indices selected")
            
            st.markdown("---")
            
            # Chart Type Selection
            st.markdown("### 📊 Chart Type")
            
            chart_type = st.radio(
                "Select chart type",
                options=['Line Chart', 'Area Chart', 'Year-over-Year Change'],
                index=0,
                key="chart_type",
                help="Choose how to visualize the selected indices"
            )
            
            st.caption(f"📈 Displaying: {chart_type}")
            
            st.markdown("---")
            
            # Anomaly Detection Controls
            st.markdown("### 🚨 Anomaly Detection")
            
            # Enable anomaly detection
            enable_anomalies = st.checkbox(
                "Enable Anomaly Detection",
                value=False,
                key="enable_anomalies",
                help="Detect unusual price movements using statistical sigma bands"
            )
            
            if enable_anomalies:
                # Sensitivity control
                sigma_level = st.slider(
                    "Detection Sensitivity (σ)",
                    min_value=1.0,
                    max_value=3.0,
                    value=2.0,
                    step=0.1,
                    key="sigma_level",
                    help="Lower values = more sensitive (more anomalies detected)"
                )
                
                # Window size for rolling calculation
                window_size = st.selectbox(
                    "Rolling Window (days)",
                    options=[30, 60, 90, 120],
                    index=1,  # Default to 60 days
                    key="window_size",
                    help="Period for calculating rolling statistics"
                )
                
                # Visualization options
                col_anom1, col_anom2 = st.columns(2)
                
                with col_anom1:
                    show_bands = st.checkbox(
                        "Show Sigma Bands",
                        value=True,
                        key="show_bands",
                        help="Display statistical bands around normal range"
                    )
                
                with col_anom2:
                    show_historical = st.checkbox(
                        "Show Crisis Periods",
                        value=True,
                        key="show_historical",
                        help="Highlight known food crisis periods"
                    )
                
                # Color settings
                with st.expander("🎨 Anomaly Colors", expanded=False):
                    col_color1, col_color2 = st.columns(2)
                    
                    with col_color1:
                        high_color = st.color_picker(
                            "High Anomalies",
                            value="#FF4444",
                            key="high_color",
                            help="Color for values above normal range"
                        )
                    
                    with col_color2:
                        low_color = st.color_picker(
                            "Low Anomalies", 
                            value="#4444FF",
                            key="low_color",
                            help="Color for values below normal range"
                        )
                
                # Create anomaly settings dict
                anomaly_settings = {
                    'enabled': True,
                    'sigma': sigma_level,
                    'window': window_size,
                    'show_bands': show_bands,
                    'show_historical': show_historical,
                    'high_color': high_color,
                    'low_color': low_color
                }
                
                # Store in session state for use in chart building
                st.session_state.anomaly_settings = anomaly_settings
                
                # Show current settings summary
                st.caption(f"🔧 Settings: {sigma_level}σ, {window_size}d window")
            else:
                # Disabled state
                st.session_state.anomaly_settings = {'enabled': False}
                st.info("💡 Enable to detect price spikes and unusual market movements")
            
            st.markdown("### 📊 Export to Excel")
            
            if df is not None and len(df) > 0 and selected_indices:
                # Get selected column names for export
                selected_columns = [index_mapping[idx] for idx in selected_indices 
                                  if idx in index_mapping and index_mapping[idx] in df.columns]
                
                if selected_columns:
                    # Generate filename with current date
                    current_date = datetime.now().strftime('%Y%m%d')
                    filename = f"FPI_Report_{current_date}.xlsx"
                    
                    @st.cache_data
                    def create_comprehensive_excel_report(data_df, selected_cols, indices_list, chart_type_param, index_mapping):
                        """Create comprehensive Excel report with data, charts, and statistics"""
                        from io import BytesIO
                        import xlsxwriter
                        
                        output = BytesIO()
                        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
                        
                        # Initialize ExcelExporter
                        exporter = ExcelExporter()
                        
                        # Prepare data with proper data types
                        export_columns = ['date'] + selected_cols
                        export_df = data_df[export_columns].copy()
                        
                        # Ensure proper data types for Excel
                        export_df['date'] = pd.to_datetime(export_df['date'])
                        for col in selected_cols:
                            export_df[col] = pd.to_numeric(export_df[col], errors='coerce')
                        
                        # Sheet 1: Data with proper formatting
                        data_workbook = exporter.generate_data_sheet(export_df, 'Data')
                        data_workbook.close()  # Close temporary workbook
                        
                        # Recreate workbook for multi-sheet report
                        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
                        
                        # Create data sheet manually with better control
                        worksheet_data = workbook.add_worksheet('Data')
                        
                        # Create formats
                        header_format = workbook.add_format({
                            'bold': True,
                            'font_color': '#FFFFFF',
                            'bg_color': '#4472C4',
                            'border': 1,
                            'align': 'center'
                        })
                        
                        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
                        number_format = workbook.add_format({'num_format': '#,##0.0'})
                        
                        # Write headers
                        for col_idx, column_name in enumerate(export_df.columns):
                            worksheet_data.write(0, col_idx, str(column_name), header_format)
                        
                        # Write data with proper formatting
                        for row_idx, (_, row) in enumerate(export_df.iterrows(), start=1):
                            for col_idx, value in enumerate(row):
                                if pd.isna(value):
                                    worksheet_data.write(row_idx, col_idx, "")
                                elif col_idx == 0:  # Date column
                                    worksheet_data.write_datetime(row_idx, col_idx, value, date_format)
                                else:  # Numeric columns
                                    worksheet_data.write_number(row_idx, col_idx, float(value), number_format)
                        
                        # Auto-adjust column widths
                        for col_idx, column_name in enumerate(export_df.columns):
                            if col_idx == 0:  # Date column
                                worksheet_data.set_column(col_idx, col_idx, 12)
                            else:
                                worksheet_data.set_column(col_idx, col_idx, 15)
                        
                        # Freeze first row
                        worksheet_data.freeze_panes(1, 0)
                        
                        # Sheet 2: Chart (try to add current chart)
                        try:
                            # Prepare chart data with proper column renaming (same as dashboard)
                            chart_data = export_df.set_index('date')[selected_cols].copy()
                            
                            # Create reverse mapping to get display names
                            reverse_mapping = {col: name for name, col in index_mapping.items() 
                                             if col in selected_cols}
                            chart_data.rename(columns=reverse_mapping, inplace=True)
                            
                            # Use display names for chart generation
                            display_names_list = list(chart_data.columns)
                            fig = build_chart(chart_data, chart_type_param, display_names_list)
                            chart_success = exporter.add_chart_sheet(workbook, fig, 'Price_Chart')
                        except Exception as e:
                            # Fallback: create a simple data summary sheet
                            chart_worksheet = workbook.add_worksheet('Price_Chart')
                            chart_worksheet.write('A1', f'Chart could not be generated: {str(e)}', 
                                                workbook.add_format({'font_color': 'red'}))
                        
                        # Sheet 3: Statistics
                        try:
                            stats_df = calculate_detailed_statistics(data_df, indices_list, index_mapping)
                            if not stats_df.empty:
                                # Remove sparkline column for Excel export
                                stats_export_df = stats_df.drop('Sparkline', axis=1) if 'Sparkline' in stats_df.columns else stats_df
                                
                                stats_worksheet = workbook.add_worksheet('Statistics')
                                
                                # Write statistics headers
                                for col_idx, column_name in enumerate(stats_export_df.columns):
                                    stats_worksheet.write(0, col_idx, str(column_name), header_format)
                                
                                # Write statistics data
                                for row_idx, (_, row) in enumerate(stats_export_df.iterrows(), start=1):
                                    for col_idx, value in enumerate(row):
                                        if pd.isna(value):
                                            stats_worksheet.write(row_idx, col_idx, "")
                                        elif isinstance(value, (int, float)) and col_idx > 0:
                                            stats_worksheet.write_number(row_idx, col_idx, float(value), number_format)
                                        else:
                                            stats_worksheet.write(row_idx, col_idx, str(value))
                                
                                # Auto-adjust statistics columns
                                for col_idx in range(len(stats_export_df.columns)):
                                    stats_worksheet.set_column(col_idx, col_idx, 15)
                                
                                stats_worksheet.freeze_panes(1, 0)
                        except Exception as e:
                            # Fallback: error message sheet
                            stats_worksheet = workbook.add_worksheet('Statistics')
                            stats_worksheet.write('A1', f'Statistics could not be generated: {str(e)}', 
                                                workbook.add_format({'font_color': 'red'}))
                        
                        workbook.close()
                        return output.getvalue()
                    
                    # Main export button
                    if st.button("📊 Export to Excel", type="primary", use_container_width=True):
                        # Show progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("🔄 Preparing data...")
                            progress_bar.progress(20)
                            
                            status_text.text("📊 Generating charts...")
                            progress_bar.progress(50)
                            
                            status_text.text("🧮 Calculating statistics...")
                            progress_bar.progress(70)
                            
                            status_text.text("📁 Creating Excel file...")
                            progress_bar.progress(90)
                            
                            # Generate comprehensive report
                            excel_data = create_comprehensive_excel_report(
                                df, selected_columns, selected_indices, chart_type, index_mapping
                            )
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Report generated successfully!")
                            
                            # Provide download button
                            st.download_button(
                                label="💾 Download FPI Report",
                                data=excel_data,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help=f"Complete report with data, charts, and statistics ({len(df)} rows, {len(selected_indices)} indices)"
                            )
                            
                            # Clear progress indicators after a moment
                            import time
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()
                            
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"Export error: {str(e)}")
                    
                    # Show export info
                    st.caption(f"📅 Export includes: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
                    st.caption(f"📊 Data points: {len(df)} | Indices: {len(selected_indices)} | Charts: {chart_type}")
                    
                else:
                    st.info("Select indices to enable export")
            else:
                st.info("Load data and select indices to enable export")
        
        # Create tabs for different analysis views
        tab1, tab2 = st.tabs(["📈 Dashboard", "🔗 Correlation Analysis"])
        
        # Tab 1: Main Dashboard
        with tab1:
            # Main content area
            col1, col2 = st.columns([2, 1])
        
            with col1:
                st.header("📈 Food Price Index Trend")
                
                if df is not None and len(df) > 0:
                    # Get selected column names from mapping
                    selected_columns = [index_mapping[idx] for idx in selected_indices 
                                      if idx in index_mapping]
                    
                    # Create chart with selected indices
                    if selected_columns:
                        # Filter DataFrame to only include date and selected columns
                        available_columns = [col for col in selected_columns if col in df.columns]
                        
                        if available_columns:
                            # Prepare data for chart
                            chart_data = df.set_index('date')[available_columns].copy()
                            
                            # Rename columns for display
                            display_names = {col: name for name, col in index_mapping.items() 
                                           if col in available_columns}
                            chart_data.rename(columns=display_names, inplace=True)
                            
                            # Build and display Plotly chart with anomaly detection
                            anomaly_settings = getattr(st.session_state, 'anomaly_settings', {'enabled': False})
                            fig = build_chart(chart_data, chart_type, list(chart_data.columns), anomaly_settings)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show data info
                        date_range = f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
                        indices_shown = len(available_columns)
                        st.caption(f"📅 Data range: {date_range} | 📊 {len(df)} data points | 📈 {indices_shown} indices displayed")
                        
                        # Anomaly Detection Insights (if enabled)
                        if anomaly_settings.get('enabled', False):
                            st.markdown("### 🚨 Anomaly Analysis")
                            
                            try:
                                # Calculate anomalies for selected indices
                                from anomaly_detector import detect_anomalies
                                
                                anomaly_summaries = {}
                                all_insights = []
                                
                                for display_name, column_name in [(name, col) for name, col in index_mapping.items() 
                                                                 if col in available_columns]:
                                    # Detect anomalies for this index
                                    df_with_anomalies = detect_anomalies(
                                        chart_data.reset_index(),
                                        display_name,
                                        window=anomaly_settings.get('window', 60),
                                        sigma=anomaly_settings.get('sigma', 2.0)
                                    )
                                    
                                    # Get summary
                                    summary = get_anomaly_summary(df_with_anomalies, display_name, 'index')
                                    anomaly_summaries[display_name] = summary
                                    
                                    # Get insights
                                    insights = get_anomaly_insights(df_with_anomalies, display_name)
                                    if insights['top_anomalies']:
                                        all_insights.extend(insights['top_anomalies'][:3])  # Top 3 per index
                                
                                # Display anomaly summary cards
                                if anomaly_summaries:
                                    anom_cols = st.columns(min(len(anomaly_summaries), 4))
                                    
                                    for i, (idx_name, summary) in enumerate(anomaly_summaries.items()):
                                        if i < len(anom_cols) and summary['total_anomalies'] > 0:
                                            with anom_cols[i]:
                                                st.metric(
                                                    label=f"🚨 {idx_name}",
                                                    value=f"{summary['total_anomalies']} anomalies",
                                                    delta=f"{summary['anomaly_rate']:.1f}% rate"
                                                )
                                                if summary['max_sigma_level'] > 0:
                                                    severity = classify_anomaly_severity(summary['max_sigma_level'])
                                                    st.caption(f"Max: {summary['max_sigma_level']:.1f}σ ({severity})")
                                
                                # Recent anomalies table
                                if all_insights:
                                    st.markdown("**📅 Recent Significant Anomalies**")
                                    
                                    # Sort by sigma level and take top 5
                                    top_anomalies = sorted(all_insights, key=lambda x: x['sigma_level'], reverse=True)[:5]
                                    
                                    anomaly_data = []
                                    for anomaly in top_anomalies:
                                        anomaly_data.append({
                                            'Date': anomaly['date'].strftime('%Y-%m-%d') if hasattr(anomaly['date'], 'strftime') else str(anomaly['date']),
                                            'Value': f"{anomaly['value']:.1f}",
                                            'Type': "📈 High" if anomaly['type'] == 'high' else "📉 Low",
                                            'Severity': f"{anomaly['sigma_level']:.1f}σ ({anomaly['severity']})"
                                        })
                                    
                                    if anomaly_data:
                                        anomaly_df = pd.DataFrame(anomaly_data)
                                        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No significant anomalies detected in the selected period.")
                                else:
                                    st.info("No anomalies detected with current settings. Try lowering the sensitivity.")
                                
                            except Exception as e:
                                st.error(f"Error in anomaly analysis: {str(e)}")
                                st.info("Anomaly detection requires sufficient historical data.")
                        
                        # KPI Cards Row below chart
                        st.markdown("### 📊 Key Performance Indicators")
                        
                        if selected_columns:
                            try:
                                # Calculate KPIs for selected indices
                                kpis = calculate_kpis(df, selected_columns)
                                
                                # Create dynamic columns based on number of selected indices
                                num_indices = len(selected_columns)
                                if num_indices <= 3:
                                    kpi_cols = st.columns(num_indices)
                                else:
                                    # For more than 3 indices, create two rows
                                    first_row_count = min(3, num_indices)
                                    kpi_cols_row1 = st.columns(first_row_count)
                                    if num_indices > 3:
                                        second_row_count = num_indices - 3
                                        kpi_cols_row2 = st.columns(second_row_count)
                                        kpi_cols = kpi_cols_row1 + kpi_cols_row2
                                    else:
                                        kpi_cols = kpi_cols_row1
                                
                                # Display KPI cards
                                col_idx = 0
                                for display_name, column_name in [(name, col) for name, col in index_mapping.items() 
                                                                 if col in selected_columns]:
                                    if column_name in kpis and col_idx < len(kpi_cols):
                                        with kpi_cols[col_idx]:
                                            kpi_data = kpis[column_name]
                                            
                                            # Current value with trend emoji
                                            trend_emoji = get_trend_emoji(kpi_data['trend_direction'])
                                            current_val = format_kpi_for_display(kpi_data['current_value'])
                                            
                                            # YoY change for delta
                                            yoy_change = kpi_data['yoy_change']
                                            yoy_display = format_kpi_for_display(yoy_change, 'percentage') if yoy_change else None
                                            delta_color = get_delta_color(yoy_change)
                                            
                                            st.metric(
                                                label=f"{trend_emoji} {display_name}",
                                                value=current_val,
                                                delta=yoy_display,
                                                delta_color=delta_color
                                            )
                                            
                                            # Show 12-month average as additional context
                                            if kpi_data['12m_avg'] is not None:
                                                avg_val = format_kpi_for_display(kpi_data['12m_avg'])
                                                st.caption(f"12-month avg: {avg_val}")
                                        
                                        col_idx += 1
                                
                            except Exception as e:
                                st.error(f"📈 Error calculating KPIs: {str(e)}")
                        else:
                            st.info("Select indices from the sidebar to see KPI metrics")
                    else:
                        st.warning("Please select at least one index to display")
                else:
                    st.error("📊 Unable to display chart - no data available")
                    st.write("""
                    This could be due to:
                    - FAO server maintenance
                    - Network connectivity issues  
                    - Data format changes
                    
                    Please try refreshing the page or check back later.
                    """)
        
            with col2:
                st.header("📊 Key Metrics")
                
                if df is not None and len(df) > 0:
                    # Get selected column names from mapping for KPI calculation
                    selected_columns = [index_mapping[idx] for idx in selected_indices 
                                      if idx in index_mapping and index_mapping[idx] in df.columns]
                    
                    if selected_columns:
                        try:
                            # Calculate KPIs for selected indices
                            kpis = calculate_kpis(df, selected_columns)
                            
                            # Display KPIs for each selected index
                            for display_name, column_name in [(name, col) for name, col in index_mapping.items() 
                                                             if col in selected_columns]:
                                if column_name in kpis:
                                    kpi_data = kpis[column_name]
                                    
                                    # Current value with trend emoji
                                    trend_emoji = get_trend_emoji(kpi_data['trend_direction'])
                                    current_val = format_kpi_for_display(kpi_data['current_value'])
                                    
                                    # YoY change for delta
                                    yoy_change = kpi_data['yoy_change']
                                    yoy_display = format_kpi_for_display(yoy_change, 'percentage') if yoy_change else None
                                    delta_color = get_delta_color(yoy_change)
                                    
                                    st.metric(
                                        label=f"{trend_emoji} {display_name}",
                                        value=current_val,
                                        delta=yoy_display,
                                        delta_color=delta_color
                                    )
                                    
                                    # Show 12-month average as additional context
                                    if kpi_data['12m_avg'] is not None:
                                        avg_val = format_kpi_for_display(kpi_data['12m_avg'])
                                        st.caption(f"12-month avg: {avg_val}")
                            
                            # Latest data date
                            latest_date = df['date'].iloc[-1].strftime('%Y-%m')
                            st.write(f"📅 **Latest data:** {latest_date}")
                            
                        except Exception as e:
                            st.error(f"📈 Error calculating KPIs: {str(e)}")
                            
                            # Fallback to basic metrics display
                            latest = df.iloc[-1]
                            st.metric(
                                label="Current Food Price Index",
                                value=f"{latest['food_price_index']:.1f}",
                                delta=None
                            )
                    else:
                        st.warning("📈 No valid indices selected for metrics")
                else:
                    st.error("📈 Unable to display metrics - no data available")
        
        # Expandable Statistics Section
        if df is not None and len(df) > 0 and selected_indices:
            with st.expander("📈 Detailed Statistics"):
                try:
                    # Calculate detailed statistics with sparklines
                    stats_df = calculate_detailed_statistics(df, selected_indices, index_mapping)
                    
                    if not stats_df.empty:
                        # Configure column display with formatting and sparklines
                        column_config = {
                            "Index": st.column_config.TextColumn(
                                "Food Price Index",
                                width="medium",
                                help="Selected food price indices"
                            ),
                            "Min": st.column_config.NumberColumn(
                                "Minimum",
                                format="%.1f",
                                help="Minimum value in selected period"
                            ),
                            "Max": st.column_config.NumberColumn(
                                "Maximum", 
                                format="%.1f",
                                help="Maximum value in selected period"
                            ),
                            "Mean": st.column_config.NumberColumn(
                                "Average",
                                format="%.1f", 
                                help="Average value in selected period"
                            ),
                            "Std Dev": st.column_config.NumberColumn(
                                "Std Deviation",
                                format="%.1f",
                                help="Standard deviation in selected period"
                            ),
                            "Sparkline": st.column_config.LineChartColumn(
                                "Trend (Last 24 months)",
                                width="large",
                                help="Recent price trend visualization"
                            )
                        }
                        
                        # Display statistics table
                        st.dataframe(
                            stats_df,
                            column_config=column_config,
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Show period information
                        period_start = df['date'].min().strftime('%Y-%m')
                        period_end = df['date'].max().strftime('%Y-%m')
                        st.caption(f"📅 Statistics period: {period_start} to {period_end} | 📊 {len(df)} data points")
                    else:
                        st.info("No statistics available for selected indices")
                        
                except Exception as e:
                    st.error(f"📊 Error calculating statistics: {str(e)}")
        
            # Interactive Pivot Analysis Section
            if df is not None and len(df) > 0 and selected_indices:
                with st.expander("📊 Interactive Pivot Analysis", expanded=False):
                    st.markdown("""
                    Create custom pivot tables to analyze food price trends across different time periods. 
                    This tool allows you to explore data relationships and identify patterns through interactive aggregation.
                    """)
                    
                    try:
                        render_pivot_interface(df, index_mapping)
                    except Exception as e:
                        st.error(f"📊 Error loading pivot interface: {str(e)}")
                        st.info("💡 This feature requires the streamlit-aggrid package. Please ensure all dependencies are installed.")
            
            # Additional sections
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.header("🌍 Commodity Breakdown")
                st.write("Detailed breakdown by commodity category (Meat, Dairy, Cereals, Oils, Sugar) will be displayed here.")
                st.info("🍖🥛🌾🫒🍯 Commodity-specific charts coming soon.")
            
            with col4:
                st.header("📋 Data Table")
                st.write("Interactive data table with filtering and sorting capabilities will be available here.")
                st.info("📄 Sortable and filterable data table will be implemented.")
        
        # Tab 2: Correlation Analysis
        with tab2:
            render_correlation_analysis(df, index_mapping, selected_indices)
        
        # Footer with timestamp (outside tabs)
        st.markdown("---")
        col_center = st.columns([1, 2, 1])[1]
        
        with col_center:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"🕒 Dashboard updated: {current_time}")
            st.caption("📊 Data source: Food and Agriculture Organization (FAO)")
            if df is not None:
                st.caption(f"✅ Data loaded successfully with {len(df)} records")
            else:
                st.caption("⚠️ Data loading issues detected")
        
    except Exception as e:
        st.error(f"❌ An error occurred while loading the dashboard: {str(e)}")
        st.write("Please refresh the page or contact support if the problem persists.")


if __name__ == "__main__":
    main()