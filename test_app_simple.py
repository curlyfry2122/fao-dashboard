#!/usr/bin/env python3
"""Simple test for the Streamlit app and pivot functionality."""

import requests
import sys
import re


def test_app_accessibility():
    """Test if the Streamlit app is accessible and returns valid content."""
    try:
        print("Testing app accessibility...")
        response = requests.get('http://localhost:8507', timeout=10)
        
        if response.status_code == 200:
            print("✅ App is accessible (HTTP 200)")
            
            # Check for basic Streamlit content
            content = response.text
            
            # Look for Streamlit indicators
            streamlit_indicators = [
                'streamlit',
                'data-testid',
                'st-',
                'stApp'
            ]
            
            found_indicators = [indicator for indicator in streamlit_indicators if indicator in content.lower()]
            
            if found_indicators:
                print(f"✅ Streamlit app detected (found: {', '.join(found_indicators)})")
                return True
            else:
                print("⚠️  Response received but doesn't appear to be a Streamlit app")
                return False
        else:
            print(f"❌ App returned status code: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Cannot connect to app: {e}")
        return False


def test_app_modules():
    """Test that app modules can be imported successfully."""
    print("Testing module imports...")
    
    modules_to_test = [
        ('app', 'Main application'),
        ('pivot_builder', 'Pivot functionality'),
        ('chart_builder', 'Chart functionality'),
        ('data_pipeline', 'Data pipeline')
    ]
    
    import_results = []
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description} module imported successfully")
            import_results.append(True)
        except ImportError as e:
            print(f"❌ {description} module import failed: {e}")
            import_results.append(False)
    
    return all(import_results)


def test_pivot_functions():
    """Test that pivot functions work correctly."""
    print("Testing pivot functions...")
    
    try:
        from pivot_builder import (
            create_pivot_table, 
            validate_pivot_size, 
            prepare_temporal_dimensions,
            render_pivot_interface
        )
        import pandas as pd
        
        # Create test data
        test_data = {
            'date': pd.date_range('2023-01-01', periods=12, freq='ME'),
            'food_price_index': range(100, 112),
            'meat': range(95, 107)
        }
        df = pd.DataFrame(test_data)
        
        index_mapping = {
            'Food Price Index': 'food_price_index',
            'Meat': 'meat'
        }
        
        # Test temporal dimensions
        df_with_dims = prepare_temporal_dimensions(df)
        if not all(col in df_with_dims.columns for col in ['Year', 'Quarter', 'Month']):
            print("❌ Temporal dimensions test failed")
            return False
        
        # Test pivot creation
        pivot_df = create_pivot_table(df, 'Year', ['Food Price Index'], 'mean', index_mapping)
        if pivot_df.empty:
            print("❌ Pivot creation test failed")
            return False
        
        # Test size validation
        is_valid, _, _ = validate_pivot_size(df, 'Year', ['Food Price Index'], 1000)
        if not is_valid:
            print("❌ Size validation test failed") 
            return False
        
        print("✅ All pivot functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Pivot function test failed: {e}")
        return False


def test_app_integration():
    """Test that the app integrates pivot functionality correctly."""
    print("Testing app integration...")
    
    try:
        # Import the main app module
        import app
        
        # Check if the app module has our imports
        app_source = open('app.py').read()
        
        if 'from pivot_builder import render_pivot_interface' in app_source:
            print("✅ Pivot builder imported in app")
        else:
            print("❌ Pivot builder not imported in app")
            return False
        
        if 'render_pivot_interface' in app_source:
            print("✅ Pivot interface called in app")
        else:
            print("❌ Pivot interface not called in app")
            return False
        
        if 'Interactive Pivot Analysis' in app_source:
            print("✅ Pivot section found in app")
        else:
            print("❌ Pivot section not found in app")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False


def main():
    """Run all simple tests."""
    print("🧪 Running simple app and pivot tests...\n")
    
    tests = [
        ("App Accessibility", test_app_accessibility),
        ("Module Imports", test_app_modules),
        ("Pivot Functions", test_pivot_functions),
        ("App Integration", test_app_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("SIMPLE TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    overall_status = "✅ SUCCESS" if passed_tests == total_tests else "⚠️  ISSUES FOUND"
    print(f"\nOverall: {overall_status} ({passed_tests}/{total_tests})")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! App and pivot interface are working!")
        return 0
    else:
        print("⚠️  Some issues found - check details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())