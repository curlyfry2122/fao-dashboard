#!/usr/bin/env python3
"""Test the JavaScript fix directly with AgGrid."""

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode


def main():
    st.title("🔧 JavaScript Fix Test")
    
    # Create test data
    df = pd.DataFrame({
        'Year': [2023, 2024],
        'Value': [120.5, 125.8]
    })
    
    st.header("1. Test Data")
    st.dataframe(df)
    
    st.header("2. AgGrid with Corrected JavaScript")
    
    # Test the corrected JavaScript formatter
    gb = GridOptionsBuilder.from_dataframe(df)
    
    # Configure with the exact same JavaScript as in pivot_builder
    gb.configure_column('Value',
                       type=['numericColumn'],
                       valueFormatter=JsCode("function(params) { return params.value != null ? params.value.toFixed(1) : ''; }"),
                       width=100)
    
    options = gb.build()
    
    st.markdown("**JavaScript Code Being Used:**")
    for col_def in options['columnDefs']:
        if 'valueFormatter' in col_def:
            formatter = col_def['valueFormatter']
            if hasattr(formatter, 'js_code'):
                js_code = formatter.js_code.replace('::JSCODE::', '')
                st.code(js_code, language='javascript')
    
    st.markdown("**AgGrid Result:**")
    
    try:
        AgGrid(df, gridOptions=options, height=200, theme='streamlit', key='js_test')
        st.success("✅ AgGrid displayed successfully - JavaScript is working!")
    except Exception as e:
        st.error(f"❌ AgGrid failed: {e}")
        
        # Alternative test with basic formatter
        st.markdown("**Trying basic number formatter:**")
        gb_simple = GridOptionsBuilder.from_dataframe(df)
        gb_simple.configure_column('Value', 
                                  type=['numericColumn'],
                                  valueFormatter=JsCode("function(params) { return params.value ? Number(params.value).toFixed(1) : ''; }"))
        simple_options = gb_simple.build()
        
        try:
            AgGrid(df, gridOptions=simple_options, height=200, theme='streamlit', key='js_simple_test')
            st.success("✅ Simple formatter works!")
        except Exception as e2:
            st.error(f"❌ Simple formatter also failed: {e2}")


if __name__ == "__main__":
    main()