"""FAO Food Price Index Dashboard using Streamlit."""

import streamlit as st
import pandas as pd
from datetime import datetime
from data_pipeline import DataPipeline


@st.cache_data(ttl=3600)  # Cache for 1 hour
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
            
            st.markdown("### Data Options")
            st.write("Options for data selection and display preferences will be available here.")
            
            st.markdown("### Export")
            st.write("Data export functionality will be implemented here.")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📈 Food Price Index Trend")
            
            if df is not None and len(df) > 0:
                # Get selected column names from mapping
                selected_columns = [index_mapping[idx] for idx in selected_indices 
                                  if idx in index_mapping]
                
                # Create line chart with selected indices
                if selected_columns:
                    # Filter DataFrame to only include date and selected columns
                    available_columns = [col for col in selected_columns if col in df.columns]
                    
                    if available_columns:
                        chart_data = df.set_index('date')[available_columns].copy()
                        
                        # Rename columns for display
                        display_names = {col: name for name, col in index_mapping.items() 
                                       if col in available_columns}
                        chart_data.rename(columns=display_names, inplace=True)
                        
                        st.line_chart(chart_data, height=400)
                        
                        # Show data info
                        date_range = f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
                        indices_shown = len(available_columns)
                        st.caption(f"📅 Data range: {date_range} | 📊 {len(df)} data points | 📈 {indices_shown} indices displayed")
                    else:
                        st.warning("Selected indices not available in the data")
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
                # Get latest data point
                latest = df.iloc[-1]
                
                # Current Food Price Index
                st.metric(
                    label="Current Food Price Index",
                    value=f"{latest['food_price_index']:.1f}",
                    delta=None
                )
                
                # Year-over-Year Change
                if pd.notna(latest['food_price_index_yoy_change']):
                    yoy_change = latest['food_price_index_yoy_change']
                    st.metric(
                        label="Year-over-Year Change",
                        value=f"{yoy_change:.1f}%",
                        delta=f"{yoy_change:.1f}%"
                    )
                
                # Month-over-Month Change
                if pd.notna(latest['food_price_index_mom_change']):
                    mom_change = latest['food_price_index_mom_change']
                    st.metric(
                        label="Month-over-Month Change",
                        value=f"{mom_change:.1f}%",
                        delta=f"{mom_change:.1f}%"
                    )
                
                # Latest data date
                latest_date = latest['date'].strftime('%Y-%m')
                st.write(f"📅 **Latest data:** {latest_date}")
                
            else:
                st.error("📈 Unable to display metrics - no data available")
        
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
        
        # Footer with timestamp
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