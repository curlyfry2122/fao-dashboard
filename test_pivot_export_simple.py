#!/usr/bin/env python3
"""
Simple test for pivot Excel export functionality.
Uses the same data format as the main app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main app to get actual data
from app import load_fao_data
from pivot_builder import export_pivot_to_excel, create_pivot_table

def test_pivot_export_with_real_data():
    """Test pivot export with real FAO data"""
    print("\n🔄 Loading real FAO data...")
    
    try:
        # Load data using the app's function
        df_annual, df_monthly = load_fao_data()
        
        if df_monthly is None or df_monthly.empty:
            print("❌ No data available")
            return False
            
        print(f"✅ Data loaded: {len(df_monthly)} monthly records")
        print(f"   Columns: {list(df_monthly.columns)}")
        print(f"   Indices: {df_monthly['index'].nunique()} unique indices")
        
        # Prepare data with lowercase column names as expected
        df_monthly_lower = df_monthly.copy()
        df_monthly_lower.columns = df_monthly_lower.columns.str.lower()
        
        # Get available indices
        available_indices = df_monthly_lower['index'].unique()[:5]  # Test with 5 indices
        index_mapping = {idx: idx for idx in available_indices}
        
        # Test different pivot configurations
        test_configs = [
            ('year', 'mean', 'Year aggregation'),
            ('month', 'mean', 'Month aggregation'),
            ('quarter', 'max', 'Quarter with max'),
        ]
        
        for row_dim, agg_func, desc in test_configs:
            print(f"\n📊 Testing: {desc}")
            print(f"   Row dimension: {row_dim}")
            print(f"   Aggregation: {agg_func}")
            print(f"   Indices: {len(available_indices)}")
            
            try:
                # Create pivot
                pivot_df = create_pivot_table(
                    df=df_monthly_lower,
                    row_dim=row_dim,
                    col_indices=list(available_indices),
                    agg_func=agg_func,
                    index_mapping=index_mapping
                )
                
                print(f"   Pivot shape: {pivot_df.shape}")
                
                # Export to Excel
                excel_data = export_pivot_to_excel(
                    pivot_df=pivot_df,
                    row_dimension=row_dim,
                    selected_indices=list(available_indices),
                    aggregation=agg_func,
                    original_df=df_monthly_lower,
                    index_mapping=index_mapping
                )
                
                # Save file
                filename = f'pivot_export_{row_dim}_{agg_func}.xlsx'
                with open(filename, 'wb') as f:
                    f.write(excel_data.getvalue())
                
                print(f"   ✅ Exported to: {filename}")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n🎉 All pivot exports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pivot_ui_integration():
    """Test the complete pivot UI workflow"""
    print("\n🖥️ Testing pivot UI integration...")
    
    try:
        # Import the render function
        from pivot_builder import render_pivot_interface
        
        # This would need to be run in a Streamlit context
        print("   ℹ️ UI integration test requires Streamlit runtime")
        print("   Run 'streamlit run app.py' and test manually")
        
        return True
        
    except Exception as e:
        print(f"❌ UI test failed: {str(e)}")
        return False

def main():
    """Run pivot export tests"""
    print("=" * 60)
    print("🧪 PIVOT EXCEL EXPORT TEST")
    print("=" * 60)
    
    # Run tests
    success = test_pivot_export_with_real_data()
    
    if success:
        print("\n✅ Pivot export functionality verified!")
        print("\n📁 Generated Excel files:")
        import glob
        for file in glob.glob("pivot_export_*.xlsx"):
            print(f"  - {file}")
        print("\n💡 Open these files to verify formatting and content")
    else:
        print("\n❌ Tests failed - review implementation")
    
    # UI integration note
    test_pivot_ui_integration()
    
    print("\n" + "=" * 60)
    print("To test the complete feature in the app:")
    print("1. Run: streamlit run app.py")
    print("2. Expand 'Interactive Pivot Analysis' section")
    print("3. Select pivot configuration")
    print("4. Click 'Export Pivot to Excel' button")
    print("5. Verify the downloaded Excel file")
    print("=" * 60)

if __name__ == "__main__":
    main()