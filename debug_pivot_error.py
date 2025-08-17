#!/usr/bin/env python3
"""Debug script to identify the source of the 'x is not defined' error."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

from pivot_builder import configure_aggrid_options, create_pivot_table
from st_aggrid import AgGrid


def create_test_data():
    """Create minimal test data for debugging."""
    return pd.DataFrame({
        'Year': [2023, 2024],
        'Food Price Index': [120.5, 125.8],
        'Meat': [115.2, 118.9]
    })


def debug_aggrid_options():
    """Debug the AgGrid options generation."""
    print("🔍 Debugging AgGrid options generation...")
    
    df = create_test_data()
    options = configure_aggrid_options(df, 'Year')
    
    print(f"Generated options keys: {options.keys()}")
    
    if 'columnDefs' in options:
        print(f"Number of column definitions: {len(options['columnDefs'])}")
        
        for i, col_def in enumerate(options['columnDefs']):
            print(f"\nColumn {i}: {col_def.get('field', 'unknown')}")
            
            if 'valueFormatter' in col_def:
                formatter = col_def['valueFormatter']
                print(f"  Value formatter: {formatter}")
                
                # Check if it's the problematic formatter
                if hasattr(formatter, 'code'):
                    js_code = formatter.code
                    print(f"  JavaScript code: {js_code}")
                    
                    if 'function(x)' not in js_code:
                        print("  ❌ ISSUE FOUND: Missing function wrapper!")
                        return False
                else:
                    print(f"  Formatter type: {type(formatter)}")
            
            if 'cellStyle' in col_def:
                print(f"  Has cell style: Yes")
    
    print("✅ AgGrid options look correct")
    return True


def test_minimal_aggrid():
    """Test AgGrid with minimal configuration."""
    print("🧪 Testing minimal AgGrid configuration...")
    
    df = create_test_data()
    
    try:
        # Test 1: Basic AgGrid without custom formatting
        print("Test 1: Basic AgGrid")
        AgGrid(df, height=200, theme='streamlit')
        print("✅ Basic AgGrid works")
        
        # Test 2: AgGrid with our configuration
        print("Test 2: AgGrid with custom configuration")
        options = configure_aggrid_options(df, 'Year')
        AgGrid(df, gridOptions=options, height=200, theme='streamlit')
        print("✅ Custom AgGrid works")
        
        return True
        
    except Exception as e:
        print(f"❌ AgGrid test failed: {e}")
        return False


def main():
    """Main debug function."""
    st.title("🔍 Pivot Error Debug Tool")
    
    st.markdown("""
    This tool helps diagnose the 'x is not defined' error in the pivot interface.
    """)
    
    # Debug section 1: Options generation
    st.header("1. AgGrid Options Debugging")
    
    if st.button("Debug AgGrid Options"):
        with st.expander("Debug Output", expanded=True):
            st.code("""
import sys
from io import StringIO

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

try:
    result = debug_aggrid_options()
    output = mystdout.getvalue()
finally:
    sys.stdout = old_stdout

st.text(output)
if result:
    st.success("✅ AgGrid options are correct")
else:
    st.error("❌ Issues found in AgGrid options")
            """)
            
            # Run the actual debug
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            try:
                result = debug_aggrid_options()
                output = mystdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            st.text(output)
            if result:
                st.success("✅ AgGrid options are correct")
            else:
                st.error("❌ Issues found in AgGrid options")
    
    # Debug section 2: Live AgGrid test
    st.header("2. Live AgGrid Test")
    
    st.markdown("Testing AgGrid components directly:")
    
    df = create_test_data()
    
    try:
        st.subheader("Basic AgGrid (no custom formatting)")
        AgGrid(df, height=150, theme='streamlit', key='basic_test')
        st.success("✅ Basic AgGrid works")
        
        st.subheader("Custom AgGrid (with formatting)")
        options = configure_aggrid_options(df, 'Year')
        AgGrid(df, gridOptions=options, height=150, theme='streamlit', key='custom_test')
        st.success("✅ Custom AgGrid works")
        
    except Exception as e:
        st.error(f"❌ AgGrid failed: {e}")
        st.code(str(e))
    
    # Debug section 3: Pivot interface test
    st.header("3. Full Pivot Interface Test")
    
    if st.button("Test Full Pivot Interface"):
        try:
            from pivot_builder import render_pivot_interface
            
            # Create more realistic test data
            test_data = {
                'date': pd.date_range('2023-01-01', periods=12, freq='ME'),
                'food_price_index': range(100, 112),
                'meat': range(95, 107),
                'dairy': range(105, 117)
            }
            test_df = pd.DataFrame(test_data)
            
            index_mapping = {
                'Food Price Index': 'food_price_index',
                'Meat': 'meat',
                'Dairy': 'dairy'
            }
            
            with st.expander("Full Pivot Interface", expanded=True):
                render_pivot_interface(test_df, index_mapping)
                
        except Exception as e:
            st.error(f"❌ Full pivot interface failed: {e}")
            st.code(str(e))
    
    # Debug section 4: Browser cache info
    st.header("4. Cache Information")
    
    st.markdown("""
    **If you're still seeing the 'x is not defined' error:**
    
    1. **Clear browser cache**: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
    2. **Hard refresh**: Ctrl+F5 (or Cmd+Shift+R on Mac)  
    3. **Open in incognito/private window**
    4. **Restart Streamlit**: Stop and restart the `streamlit run` command
    
    The error might be due to browser caching the old JavaScript code.
    """)
    
    # JavaScript version check
    st.subheader("Current JavaScript Code")
    with st.expander("View Current JavaScript Code"):
        import inspect
        from pivot_builder import configure_aggrid_options
        
        # Get the source code
        source = inspect.getsource(configure_aggrid_options)
        st.code(source, language='python')


if __name__ == "__main__":
    main()