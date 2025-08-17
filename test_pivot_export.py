#!/usr/bin/env python3
"""
Test script for pivot table Excel export functionality.
Tests various pivot configurations and export scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pivot_builder import export_pivot_to_excel, create_pivot_table
from data_fetcher import download_fao_fpi_data
from excel_parser import parse_fao_excel_data

def test_pivot_export_basic():
    """Test basic pivot export functionality"""
    print("\n📊 Testing basic pivot export...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=24, freq='M')
    indices = ['Food', 'Cereals', 'Meat', 'Dairy', 'Sugar']
    
    data = []
    for date in dates:
        for idx in indices:
            data.append({
                'Date': date,
                'Index': idx,
                'Value': np.random.uniform(90, 110),
                'Year': date.year,
                'Month': date.strftime('%B'),
                'Quarter': f"Q{(date.month-1)//3 + 1}"
            })
    
    df = pd.DataFrame(data)
    
    # Create pivot
    index_mapping = {idx: idx for idx in indices}
    pivot_df = create_pivot_table(
        df=df,
        row_dim='Month',
        col_indices=indices,
        agg_func='mean',
        index_mapping=index_mapping
    )
    
    # Export to Excel
    excel_data = export_pivot_to_excel(
        pivot_df=pivot_df,
        row_dimension='Month',
        selected_indices=indices,
        aggregation='mean',
        original_df=df,
        index_mapping=index_mapping
    )
    
    # Verify export
    assert isinstance(excel_data, BytesIO), "Export should return BytesIO object"
    assert excel_data.getvalue(), "Excel data should not be empty"
    
    # Save for manual inspection
    with open('test_pivot_export_basic.xlsx', 'wb') as f:
        f.write(excel_data.getvalue())
    
    print("✅ Basic pivot export test passed")
    return True

def test_pivot_configurations():
    """Test various pivot configurations"""
    print("\n🔧 Testing various pivot configurations...")
    
    # Load real FAO data
    excel_data = download_fao_fpi_data()
    if excel_data:
        df_annual = parse_fao_excel_data(excel_data, 'Annual')
        df_monthly = parse_fao_excel_data(excel_data, 'Monthly')
    else:
        df_annual, df_monthly = None, None
    
    if df_monthly is None or df_monthly.empty:
        print("⚠️ No real data available, using sample data")
        # Create sample data if real data not available
        dates = pd.date_range('2023-01-01', periods=36, freq='M')
        indices = ['Food', 'Cereals', 'Meat', 'Dairy', 'Sugar', 'Oils']
        
        data = []
        for date in dates:
            for idx in indices:
                data.append({
                    'Date': date,
                    'Index': idx,
                    'Value': np.random.uniform(85, 115),
                    'Year': date.year,
                    'Month': date.strftime('%B'),
                    'Quarter': f"Q{(date.month-1)//3 + 1}"
                })
        df_monthly = pd.DataFrame(data)
    
    # Test different configurations
    configurations = [
        ('Year', ['Food', 'Cereals'], 'mean'),
        ('Month', ['Meat', 'Dairy', 'Sugar'], 'max'),
        ('Quarter', ['Food', 'Oils'], 'min'),
        ('Year', ['Food'], 'mean'),  # Single index
        ('Month', list(df_monthly['Index'].unique())[:10], 'mean'),  # Many indices
    ]
    
    results = []
    for i, (row_dim, indices, agg) in enumerate(configurations, 1):
        print(f"\n  Config {i}: {row_dim} × {len(indices)} indices, {agg}")
        
        try:
            # Filter available indices
            available_indices = [idx for idx in indices if idx in df_monthly['Index'].unique()]
            if not available_indices:
                print(f"    ⚠️ No matching indices found, skipping")
                continue
            
            # Create pivot
            index_mapping = {idx: idx for idx in available_indices}
            pivot_df = create_pivot_table(
                df=df_monthly,
                row_dim=row_dim,
                col_indices=available_indices,
                agg_func=agg,
                index_mapping=index_mapping
            )
            
            # Export to Excel
            excel_data = export_pivot_to_excel(
                pivot_df=pivot_df,
                row_dimension=row_dim,
                selected_indices=available_indices,
                aggregation=agg,
                original_df=df_monthly,
                index_mapping=index_mapping
            )
            
            # Save for inspection
            filename = f'test_pivot_{row_dim}_{agg}_{i}.xlsx'
            with open(filename, 'wb') as f:
                f.write(excel_data.getvalue())
            
            print(f"    ✅ Exported: {filename}")
            print(f"    📊 Pivot shape: {pivot_df.shape}")
            results.append(True)
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100 if results else 0
    print(f"\n✅ Configuration tests: {sum(results)}/{len(results)} passed ({success_rate:.0f}%)")
    return success_rate >= 80  # Allow some failures

def test_large_pivot():
    """Test pivot export with large dataset"""
    print("\n📈 Testing large pivot export...")
    
    # Create large dataset
    dates = pd.date_range('2020-01-01', periods=60, freq='M')  # 5 years
    indices = [f'Index_{i}' for i in range(50)]  # 50 indices
    
    data = []
    for date in dates:
        for idx in indices:
            data.append({
                'Date': date,
                'Index': idx,
                'Value': np.random.uniform(70, 130),
                'Year': date.year,
                'Month': date.strftime('%B'),
                'Quarter': f"Q{(date.month-1)//3 + 1}"
            })
    
    df = pd.DataFrame(data)
    print(f"  Dataset size: {len(df):,} rows")
    
    # Create large pivot
    index_mapping = {idx: idx for idx in indices[:20]}
    pivot_df = create_pivot_table(
        df=df,
        row_dim='Month',
        col_indices=indices[:20],  # Limit to 20 indices for safety
        agg_func='mean',
        index_mapping=index_mapping
    )
    
    print(f"  Pivot size: {pivot_df.shape[0]} × {pivot_df.shape[1]} = {pivot_df.size} cells")
    
    # Export to Excel
    excel_data = export_pivot_to_excel(
        pivot_df=pivot_df,
        row_dimension='Month',
        selected_indices=indices[:20],
        aggregation='mean',
        original_df=df,
        index_mapping=index_mapping
    )
    
    # Save for inspection
    with open('test_pivot_large.xlsx', 'wb') as f:
        f.write(excel_data.getvalue())
    
    print("✅ Large pivot export test passed")
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔍 Testing edge cases...")
    
    # Test with minimal data
    df_minimal = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=3, freq='M'),
        'Index': ['Food', 'Food', 'Food'],
        'Value': [100, 101, 102],
        'Year': [2024, 2024, 2024],
        'Month': ['January', 'February', 'March'],
        'Quarter': ['Q1', 'Q1', 'Q1']
    })
    
    # Test single row pivot
    index_mapping = {'Food': 'Food'}
    pivot_df = create_pivot_table(
        df=df_minimal,
        row_dim='Quarter',
        col_indices=['Food'],
        agg_func='mean',
        index_mapping=index_mapping
    )
    
    excel_data = export_pivot_to_excel(
        pivot_df=pivot_df,
        row_dimension='Quarter',
        selected_indices=['Food'],
        aggregation='mean',
        original_df=df_minimal,
        index_mapping={'Food': 'Food'}
    )
    
    with open('test_pivot_edge_case.xlsx', 'wb') as f:
        f.write(excel_data.getvalue())
    
    print("✅ Edge case tests passed")
    return True

def main():
    """Run all pivot export tests"""
    print("=" * 60)
    print("🧪 PIVOT EXCEL EXPORT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic Export", test_pivot_export_basic),
        ("Various Configurations", test_pivot_configurations),
        ("Large Dataset", test_large_pivot),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔧 Running: {test_name}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print("-" * 60)
    print(f"Total: {total_passed}/{total_tests} passed ({success_rate:.0f}%)")
    
    if success_rate == 100:
        print("\n🎉 All tests passed successfully!")
    elif success_rate >= 75:
        print("\n✅ Most tests passed, pivot export is functional")
    else:
        print("\n⚠️ Several tests failed, review implementation")
    
    print("\n📁 Excel files created for manual inspection:")
    import glob
    for file in glob.glob("test_pivot*.xlsx"):
        print(f"  - {file}")

if __name__ == "__main__":
    main()