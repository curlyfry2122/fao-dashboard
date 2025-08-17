#!/usr/bin/env python3
"""
Simple test for anomaly detection functionality without importing the full app.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly_detector import (
    detect_anomalies,
    get_anomaly_summary,
    classify_anomaly_severity
)
from data_fetcher import download_fao_fpi_data
from excel_parser import parse_fao_excel_data

def create_test_data():
    """Create test data with known anomalies"""
    dates = pd.date_range('2020-01-01', periods=100, freq='MS')
    
    # Create normal data with some spikes
    np.random.seed(42)
    base_values = 100 + np.random.normal(0, 5, len(dates))
    
    # Add some artificial spikes (anomalies)
    spike_indices = [10, 25, 60, 85]  # Known spike positions
    for idx in spike_indices:
        if idx < len(base_values):
            base_values[idx] += 30  # Add large positive spike
    
    # Add some drops
    drop_indices = [15, 40, 70]
    for idx in drop_indices:
        if idx < len(base_values):
            base_values[idx] -= 25  # Add large negative drop
    
    df = pd.DataFrame({
        'date': dates,
        'test_index': base_values
    })
    
    return df, spike_indices, drop_indices

def test_basic_anomaly_detection():
    """Test basic anomaly detection functionality"""
    print("🧪 Testing Basic Anomaly Detection...")
    
    df, spike_indices, drop_indices = create_test_data()
    
    # Detect anomalies
    df_anomalies = detect_anomalies(df, 'test_index', sigma=2.0, window=30)
    
    # Check if we detected anomalies
    anomalies = df_anomalies[df_anomalies['test_index_is_anomaly']]
    
    print(f"   Data points: {len(df)}")
    print(f"   Anomalies detected: {len(anomalies)}")
    print(f"   Expected spikes at indices: {spike_indices}")
    print(f"   Expected drops at indices: {drop_indices}")
    
    # Check if we detected the artificial spikes
    detected_spikes = 0
    for spike_idx in spike_indices:
        # Look for anomalies near the spike
        window_start = max(0, spike_idx - 2)
        window_end = min(len(df), spike_idx + 3)
        
        window_anomalies = anomalies[
            (anomalies.index >= window_start) & 
            (anomalies.index <= window_end) &
            (anomalies['test_index_anomaly_type'] == 'high')
        ]
        
        if not window_anomalies.empty:
            detected_spikes += 1
            max_sigma = window_anomalies['test_index_sigma_level'].max()
            print(f"   ✅ Spike near index {spike_idx}: {max_sigma:.1f}σ")
        else:
            print(f"   ❌ Missed spike at index {spike_idx}")
    
    # Check drops
    detected_drops = 0
    for drop_idx in drop_indices:
        window_start = max(0, drop_idx - 2)
        window_end = min(len(df), drop_idx + 3)
        
        window_anomalies = anomalies[
            (anomalies.index >= window_start) & 
            (anomalies.index <= window_end) &
            (anomalies['test_index_anomaly_type'] == 'low')
        ]
        
        if not window_anomalies.empty:
            detected_drops += 1
            max_sigma = window_anomalies['test_index_sigma_level'].max()
            print(f"   ✅ Drop near index {drop_idx}: {max_sigma:.1f}σ")
        else:
            print(f"   ❌ Missed drop at index {drop_idx}")
    
    # Calculate success rate
    expected_anomalies = len(spike_indices) + len(drop_indices)
    detected_anomalies = detected_spikes + detected_drops
    success_rate = (detected_anomalies / expected_anomalies * 100) if expected_anomalies > 0 else 0
    
    print(f"   Detection rate: {detected_anomalies}/{expected_anomalies} ({success_rate:.0f}%)")
    
    return success_rate >= 70  # At least 70% detection rate

def test_real_fao_data():
    """Test with real FAO data"""
    print("\n📊 Testing with Real FAO Data...")
    
    try:
        # Download real data
        excel_data = download_fao_fpi_data()
        if not excel_data:
            print("   ❌ Could not download FAO data")
            return False
        
        # Parse data
        df_monthly = parse_fao_excel_data(excel_data, 'Monthly')
        if df_monthly is None or df_monthly.empty:
            print("   ❌ Could not parse FAO data")
            return False
        
        print(f"   ✅ Loaded {len(df_monthly)} monthly records")
        
        # Test detection on food price index
        if 'food_price_index' not in df_monthly.columns:
            print("   ❌ food_price_index column not found")
            return False
        
        # Detect anomalies with different sensitivities
        sensitivities = [1.5, 2.0, 2.5]
        
        for sigma in sensitivities:
            df_anomalies = detect_anomalies(df_monthly, 'food_price_index', sigma=sigma)
            summary = get_anomaly_summary(df_anomalies, 'food_price_index')
            
            print(f"   σ={sigma}: {summary['total_anomalies']} anomalies ({summary['anomaly_rate']:.1f}%)")
        
        # Look for anomalies in known crisis periods
        crisis_periods = [
            ('2008-01-01', '2008-12-31', '2008 Food Crisis'),
            ('2020-03-01', '2020-12-31', 'COVID-19 Impact'),
            ('2022-02-01', '2023-12-31', 'Ukraine Crisis')
        ]
        
        df_anomalies = detect_anomalies(df_monthly, 'food_price_index', sigma=2.0)
        
        for start_date, end_date, period_name in crisis_periods:
            period_start = pd.to_datetime(start_date)
            period_end = pd.to_datetime(end_date)
            
            period_mask = (
                (df_anomalies['date'] >= period_start) & 
                (df_anomalies['date'] <= period_end) &
                (df_anomalies['food_price_index_is_anomaly'] == True)
            )
            
            period_anomalies = df_anomalies[period_mask]
            
            if not period_anomalies.empty:
                max_sigma = period_anomalies['food_price_index_sigma_level'].max()
                count = len(period_anomalies)
                print(f"   ✅ {period_name}: {count} anomalies (max {max_sigma:.1f}σ)")
            else:
                print(f"   ⚠️ {period_name}: No anomalies detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    """Run anomaly detection tests"""
    print("=" * 50)
    print("🚨 ANOMALY DETECTION TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Detection", test_basic_anomaly_detection),
        ("Real FAO Data", test_real_fao_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal: {total_passed}/{total_tests} ({success_rate:.0f}%)")
    
    if success_rate >= 50:
        print("\n🎉 Anomaly detection is functional!")
        print("Ready to test in the Streamlit dashboard.")
    else:
        print("\n⚠️ Issues detected in anomaly detection.")

if __name__ == "__main__":
    main()