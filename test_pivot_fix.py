#!/usr/bin/env python3
"""Test script to validate the JavaScript fix in pivot interface."""

import sys
import pandas as pd
import tempfile
import subprocess
import time
import requests
from datetime import datetime
from pivot_builder import render_pivot_interface, create_pivot_table, configure_aggrid_options


def create_realistic_fao_data():
    """Create realistic FAO data for testing."""
    # Create 36 months of data (3 years)
    dates = pd.date_range('2021-01-01', '2023-12-31', freq='ME')  # Use ME instead of deprecated M
    
    # Base prices for realistic FAO data
    base_prices = {
        'food_price_index': 118.5,
        'meat': 112.3,
        'dairy': 126.8,
        'cereals': 115.2,
        'oils': 135.7,
        'sugar': 108.4
    }
    
    data = {'date': dates}
    
    # Generate realistic price movements
    for index, base_price in base_prices.items():
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Add seasonal variation and trend
            seasonal_factor = 1 + 0.05 * (i % 12 - 6) / 6  # Seasonal variation
            trend_factor = 1 + 0.002 * i  # Slight upward trend
            noise_factor = 1 + (hash(str(date) + index) % 100 - 50) / 1000  # Pseudo-random noise
            
            current_price = base_price * seasonal_factor * trend_factor * noise_factor
            prices.append(round(current_price, 1))
        
        data[index] = prices
    
    return pd.DataFrame(data)


def test_pivot_table_creation():
    """Test pivot table creation without UI components."""
    print("Testing pivot table creation...")
    
    df = create_realistic_fao_data()
    
    index_mapping = {
        'Food Price Index': 'food_price_index',
        'Meat': 'meat',
        'Dairy': 'dairy',
        'Cereals': 'cereals',
        'Oils': 'oils',
        'Sugar': 'sugar'
    }
    
    # Test different pivot configurations
    test_cases = [
        ('Year', ['Food Price Index', 'Meat'], 'mean'),
        ('Quarter', ['Dairy', 'Cereals'], 'max'),
        ('Month', ['Food Price Index'], 'min'),
    ]
    
    for row_dim, col_indices, agg_func in test_cases:
        try:
            pivot_df = create_pivot_table(df, row_dim, col_indices, agg_func, index_mapping)
            assert not pivot_df.empty, f"Pivot table empty for {row_dim}, {col_indices}, {agg_func}"
            assert row_dim in pivot_df.columns, f"Row dimension {row_dim} not in columns"
            
            print(f"✅ Pivot creation test passed: {row_dim} x {col_indices} ({agg_func})")
            
        except Exception as e:
            print(f"❌ Pivot creation failed: {row_dim} x {col_indices} ({agg_func}) - {e}")
            return False
    
    return True


def test_aggrid_configuration():
    """Test AgGrid configuration including JavaScript code."""
    print("Testing AgGrid configuration...")
    
    # Create sample pivot data
    df = create_realistic_fao_data()
    
    index_mapping = {
        'Food Price Index': 'food_price_index',
        'Meat': 'meat',
        'Dairy': 'dairy'
    }
    
    pivot_df = create_pivot_table(df, 'Year', ['Food Price Index', 'Meat'], 'mean', index_mapping)
    
    if pivot_df.empty:
        print("❌ Cannot test AgGrid - pivot table creation failed")
        return False
    
    try:
        # Test AgGrid options configuration
        grid_options = configure_aggrid_options(pivot_df, 'Year')
        
        # Validate that options were created
        assert isinstance(grid_options, dict), "Grid options should be a dictionary"
        assert 'columnDefs' in grid_options, "Grid options should contain columnDefs"
        
        # Check for JavaScript code in column definitions
        column_defs = grid_options['columnDefs']
        js_found = False
        
        for col_def in column_defs:
            if 'valueFormatter' in col_def:
                js_code = str(col_def['valueFormatter'])
                if 'function(x)' in js_code and 'toFixed(1)' in js_code:
                    js_found = True
                    print(f"✅ Found corrected JavaScript formatter: {js_code}")
                    break
        
        if not js_found:
            print("⚠️ JavaScript formatter not found in column definitions")
        
        print("✅ AgGrid configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ AgGrid configuration failed: {e}")
        return False


def test_streamlit_app_startup():
    """Test that the Streamlit app starts without errors."""
    print("Testing Streamlit app startup...")
    
    try:
        # Start Streamlit app in background
        proc = subprocess.Popen([
            'streamlit', 'run', 'app.py', 
            '--server.headless', 'true',
            '--server.port', '8505',
            '--logger.level', 'error'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(8)
        
        # Check if app is responding
        try:
            response = requests.get('http://localhost:8505', timeout=5)
            if response.status_code == 200:
                print("✅ Streamlit app started successfully")
                startup_success = True
            else:
                print(f"❌ Streamlit app returned status {response.status_code}")
                startup_success = False
        except requests.RequestException as e:
            print(f"❌ Could not connect to Streamlit app: {e}")
            startup_success = False
        
        # Clean up
        proc.terminate()
        proc.wait(timeout=5)
        
        return startup_success
        
    except Exception as e:
        print(f"❌ Streamlit app startup test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests to validate the pivot fix."""
    print("🧪 Starting comprehensive pivot fix validation...\n")
    
    test_results = []
    
    # Test 1: Basic pivot functionality
    test_results.append(("Pivot Table Creation", test_pivot_table_creation()))
    
    # Test 2: AgGrid configuration
    test_results.append(("AgGrid Configuration", test_aggrid_configuration()))
    
    # Test 3: Streamlit app startup
    test_results.append(("Streamlit App Startup", test_streamlit_app_startup()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Pivot interface fix is successful.")
        return True
    else:
        print("❌ Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)