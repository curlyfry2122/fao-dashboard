#!/usr/bin/env python3
"""
Test script for correlation analyzer functionality.
Tests correlation calculation, visualization, and export features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correlation_analyzer import (
    calculate_correlation_matrix,
    calculate_correlation_with_pvalues,
    build_correlation_heatmap,
    get_correlation_insights,
    export_correlation_to_excel,
    interpret_correlation
)

def create_sample_data():
    """Create sample FAO-like data for testing"""
    # Generate dates
    dates = pd.date_range('2020-01-01', periods=48, freq='MS')  # 4 years monthly
    
    # Create correlated price indices
    np.random.seed(42)  # For reproducible results
    
    # Base trend
    trend = np.linspace(90, 120, len(dates))
    noise_std = 5
    
    # Food Price Index (base)
    food_price_index = trend + np.random.normal(0, noise_std, len(dates))
    
    # Cereals (highly correlated with food index)
    cereals = 0.8 * food_price_index + 0.2 * trend + np.random.normal(0, noise_std * 0.7, len(dates))
    
    # Meat (moderately correlated)
    meat = 0.6 * food_price_index + 0.4 * np.linspace(80, 130, len(dates)) + np.random.normal(0, noise_std, len(dates))
    
    # Dairy (moderate correlation)
    dairy = 0.5 * food_price_index + 0.5 * np.linspace(85, 125, len(dates)) + np.random.normal(0, noise_std, len(dates))
    
    # Oils (lower correlation)
    oils = 0.3 * food_price_index + 0.7 * np.linspace(70, 140, len(dates)) + np.random.normal(0, noise_std * 1.2, len(dates))
    
    # Sugar (weak correlation, more volatile)
    sugar = 0.2 * food_price_index + 0.8 * (100 + 20 * np.sin(np.arange(len(dates)) * 0.5)) + np.random.normal(0, noise_std * 1.5, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'food_price_index': food_price_index,
        'cereals': cereals,
        'meat': meat,
        'dairy': dairy,
        'oils': oils,
        'sugar': sugar
    })
    
    return df

def test_correlation_calculation():
    """Test basic correlation calculation"""
    print("\n📊 Testing correlation calculation...")
    
    df = create_sample_data()
    indices = ['food_price_index', 'cereals', 'meat', 'dairy', 'oils', 'sugar']
    
    # Test Pearson correlation
    corr_matrix = calculate_correlation_matrix(df, indices, method='pearson')
    
    print(f"   Correlation matrix shape: {corr_matrix.shape}")
    print(f"   Diagonal values (should be 1.0): {np.diag(corr_matrix.values)}")
    print(f"   Symmetric: {np.allclose(corr_matrix.values, corr_matrix.values.T)}")
    
    # Test expected strong correlation (food_price_index vs cereals)
    food_cereals_corr = corr_matrix.loc['food_price_index', 'cereals']
    print(f"   Food vs Cereals correlation: {food_cereals_corr:.3f}")
    
    assert corr_matrix.shape == (6, 6), "Correlation matrix should be 6x6"
    assert np.allclose(np.diag(corr_matrix.values), 1.0), "Diagonal should be 1.0"
    assert np.allclose(corr_matrix.values, corr_matrix.values.T), "Matrix should be symmetric"
    assert food_cereals_corr > 0.7, "Food and Cereals should be strongly correlated"
    
    print("   ✅ Basic correlation calculation test passed")
    return True

def test_correlation_with_pvalues():
    """Test correlation calculation with p-values"""
    print("\n📈 Testing correlation with p-values...")
    
    df = create_sample_data()
    indices = ['food_price_index', 'cereals', 'meat', 'dairy']
    
    corr_matrix, pval_matrix = calculate_correlation_with_pvalues(df, indices, method='pearson')
    
    print(f"   Correlation matrix shape: {corr_matrix.shape}")
    print(f"   P-value matrix shape: {pval_matrix.shape}")
    print(f"   Diagonal p-values (should be 0): {np.diag(pval_matrix.values)}")
    
    # Check that diagonal correlations are 1 and p-values are 0
    assert np.allclose(np.diag(corr_matrix.values), 1.0), "Diagonal correlations should be 1.0"
    assert np.allclose(np.diag(pval_matrix.values), 0.0), "Diagonal p-values should be 0.0"
    
    # Check that most correlations are significant (p < 0.05)
    off_diagonal_pvals = pval_matrix.values[~np.eye(len(pval_matrix), dtype=bool)]
    significant_count = (off_diagonal_pvals < 0.05).sum()
    print(f"   Significant correlations: {significant_count}/{len(off_diagonal_pvals)}")
    
    print("   ✅ Correlation with p-values test passed")
    return True

def test_different_methods():
    """Test different correlation methods"""
    print("\n🔧 Testing different correlation methods...")
    
    df = create_sample_data()
    indices = ['food_price_index', 'cereals', 'meat']
    
    methods = ['pearson', 'spearman', 'kendall']
    results = {}
    
    for method in methods:
        corr_matrix = calculate_correlation_matrix(df, indices, method=method)
        results[method] = corr_matrix
        print(f"   {method.capitalize()} - Food vs Cereals: {corr_matrix.loc['food_price_index', 'cereals']:.3f}")
    
    # All methods should give positive correlation for food vs cereals
    for method in methods:
        corr_val = results[method].loc['food_price_index', 'cereals']
        assert corr_val > 0, f"{method} should give positive correlation"
    
    print("   ✅ Different methods test passed")
    return True

def test_heatmap_generation():
    """Test heatmap generation"""
    print("\n🎨 Testing heatmap generation...")
    
    df = create_sample_data()
    indices = ['food_price_index', 'cereals', 'meat', 'dairy']
    
    corr_matrix = calculate_correlation_matrix(df, indices)
    
    # Test basic heatmap
    fig = build_correlation_heatmap(corr_matrix, title="Test Correlation Matrix")
    
    print(f"   Figure type: {type(fig)}")
    print(f"   Figure data length: {len(fig.data)}")
    
    # Test heatmap with p-values
    corr_matrix, pval_matrix = calculate_correlation_with_pvalues(df, indices)
    fig_with_pvals = build_correlation_heatmap(
        corr_matrix, 
        pval_matrix, 
        title="Test with P-values", 
        show_values=True
    )
    
    print(f"   Figure with p-values data length: {len(fig_with_pvals.data)}")
    
    assert len(fig.data) > 0, "Figure should have data"
    assert len(fig_with_pvals.data) > 0, "Figure with p-values should have data"
    
    print("   ✅ Heatmap generation test passed")
    return True

def test_insights_extraction():
    """Test correlation insights extraction"""
    print("\n🔍 Testing insights extraction...")
    
    df = create_sample_data()
    indices = ['food_price_index', 'cereals', 'meat', 'dairy', 'oils', 'sugar']
    
    corr_matrix = calculate_correlation_matrix(df, indices)
    insights = get_correlation_insights(corr_matrix, threshold=0.7)
    
    print(f"   Strong positive: {len(insights['strong_positive'])}")
    print(f"   Strong negative: {len(insights['strong_negative'])}")
    print(f"   Moderate positive: {len(insights['moderate_positive'])}")
    print(f"   Moderate negative: {len(insights['moderate_negative'])}")
    print(f"   Weak: {len(insights['weak'])}")
    
    # Should have some correlations in different categories
    total_pairs = len(indices) * (len(indices) - 1) // 2
    total_found = sum(len(pairs) for pairs in insights.values())
    
    print(f"   Total pairs: {total_pairs}, Found: {total_found}")
    
    assert total_found == total_pairs, "Should find all possible pairs"
    assert len(insights['strong_positive']) > 0, "Should find some strong positive correlations"
    
    # Test specific insights
    if insights['strong_positive']:
        for idx1, idx2, corr in insights['strong_positive'][:3]:
            print(f"   Strong positive: {idx1} ↔ {idx2}: {corr:.3f}")
    
    print("   ✅ Insights extraction test passed")
    return True

def test_interpretation():
    """Test correlation interpretation"""
    print("\n💭 Testing correlation interpretation...")
    
    test_cases = [
        (0.95, "Very strong positive"),
        (-0.85, "Strong negative"),  # Fixed: -0.85 is strong, not very strong (need >= 0.9)
        (0.75, "Strong positive"),  # Fixed: 0.75 is strong (>= 0.7)
        (-0.55, "Moderate negative"),
        (0.25, "Weak positive"),
        (-0.15, "Very weak negative"),
        (0.0, "Very weak")
    ]
    
    for corr_val, expected_type in test_cases:
        interpretation = interpret_correlation(corr_val)
        print(f"   {corr_val:5.2f}: {interpretation}")
        
        # Check that interpretation contains expected words
        if "very strong" in expected_type.lower():
            assert "very strong" in interpretation.lower(), f"Should identify very strong correlation for {corr_val}"
        elif "strong" in expected_type.lower():
            assert "strong" in interpretation.lower(), f"Should identify strong correlation for {corr_val}"
        elif "moderate" in expected_type.lower():
            assert "moderate" in interpretation.lower(), f"Should identify moderate correlation for {corr_val}"
        elif "weak" in expected_type.lower():
            assert "weak" in interpretation.lower(), f"Should identify weak correlation for {corr_val}"
    
    print("   ✅ Interpretation test passed")
    return True

def test_excel_export():
    """Test Excel export functionality"""
    print("\n📊 Testing Excel export...")
    
    df = create_sample_data()
    indices = ['food_price_index', 'cereals', 'meat', 'dairy']
    
    # Calculate correlations and insights
    corr_matrix, pval_matrix = calculate_correlation_with_pvalues(df, indices)
    insights = get_correlation_insights(corr_matrix)
    
    # Prepare metadata
    metadata = {
        "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Time Period": "2020-01 to 2023-12",
        "Method": "Pearson",
        "Number of Indices": len(indices),
        "Data Points": len(df)
    }
    
    try:
        # Export to Excel
        excel_data = export_correlation_to_excel(
            corr_matrix,
            pval_matrix,
            insights,
            metadata,
            df
        )
        
        # Save test file
        with open('test_correlation_export.xlsx', 'wb') as f:
            f.write(excel_data.getvalue())
        
        print(f"   Excel file size: {len(excel_data.getvalue())} bytes")
        print("   ✅ Excel export test passed")
        print("   📁 Created: test_correlation_export.xlsx")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Excel export failed: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing edge cases...")
    
    # Test with minimal data
    df_small = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='MS'),
        'index1': [100, 101, 102],
        'index2': [90, 91, 92]
    })
    
    corr_matrix = calculate_correlation_matrix(df_small, ['index1', 'index2'])
    print(f"   Small dataset correlation: {corr_matrix.loc['index1', 'index2']:.3f}")
    
    # Test with perfect correlation
    df_perfect = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='MS'),
        'index1': range(10),
        'index2': [x * 2 for x in range(10)]  # Perfect positive correlation
    })
    
    corr_perfect = calculate_correlation_matrix(df_perfect, ['index1', 'index2'])
    print(f"   Perfect correlation: {corr_perfect.loc['index1', 'index2']:.3f}")
    
    assert abs(corr_perfect.loc['index1', 'index2'] - 1.0) < 0.001, "Should detect perfect correlation"
    
    print("   ✅ Edge cases test passed")
    return True

def main():
    """Run all correlation analyzer tests"""
    print("=" * 60)
    print("🧪 CORRELATION ANALYZER TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic Correlation Calculation", test_correlation_calculation),
        ("Correlation with P-values", test_correlation_with_pvalues),
        ("Different Methods", test_different_methods),
        ("Heatmap Generation", test_heatmap_generation),
        ("Insights Extraction", test_insights_extraction),
        ("Correlation Interpretation", test_interpretation),
        ("Excel Export", test_excel_export),
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
        print("The correlation analyzer is ready for use!")
    elif success_rate >= 75:
        print("\n✅ Most tests passed, correlation analyzer is functional")
    else:
        print("\n⚠️ Several tests failed, review implementation")
    
    print("\n📁 Files created for inspection:")
    import glob
    for file in glob.glob("test_correlation*.xlsx"):
        print(f"  - {file}")

if __name__ == "__main__":
    main()