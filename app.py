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
        
        # Sidebar
        with st.sidebar:
            st.header("📋 Dashboard Controls")
            st.markdown("### Filters & Settings")
            st.write("Interactive filters and controls will be added here to customize the data view.")
            
            st.markdown("### Data Options")
            st.write("Options for data selection and display preferences will be available here.")
            
            st.markdown("### Export")
            st.write("Data export functionality will be implemented here.")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📈 Food Price Index Trend")
            
            if df is not None and len(df) > 0:
                # Create line chart of food price index
                chart_data = df.set_index('date')[['food_price_index']].copy()
                chart_data.columns = ['Food Price Index']
                
                st.line_chart(chart_data, height=400)
                
                # Show data info
                date_range = f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
                st.caption(f"📅 Data range: {date_range} | 📊 {len(df)} data points")
                
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