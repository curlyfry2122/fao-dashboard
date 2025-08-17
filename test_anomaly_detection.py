#!/usr/bin/env python3
"""
Test script for anomaly detection functionality.
Tests detection of historical food price spikes and crisis periods.
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
    get_anomaly_insights,
    identify_historical_periods,
    classify_anomaly_severity
)
from app import load_fao_data

def test_historical_spike_detection():
    """Test anomaly detection on historical food crisis periods"""
    print("\n🕰️ Testing Historical Spike Detection...")
    
    try:
        # Load real FAO data
        df, error = load_fao_data()
        
        if error or df is None or df.empty:
            print("❌ Could not load FAO data for testing")
            return False
        
        print(f"✅ Data loaded: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
        
        # Test different sigma levels on food price index
        test_configs = [
            (1.5, "Very Sensitive"),
            (2.0, "Standard"),  
            (2.5, "Conservative"),
            (3.0, "Very Conservative")
        ]
        
        for sigma, description in test_configs:
            print(f"\n📊 Testing {description} Detection (σ={sigma})")
            
            # Detect anomalies in food price index
            df_anomalies = detect_anomalies(df, 'food_price_index', sigma=sigma)
            
            # Get summary
            summary = get_anomaly_summary(df_anomalies, 'food_price_index')
            
            print(f"   Total anomalies: {summary['total_anomalies']}")
            print(f"   High anomalies: {summary['high_anomalies']}")
            print(f"   Low anomalies: {summary['low_anomalies']}")
            print(f"   Anomaly rate: {summary['anomaly_rate']:.1f}%")
            print(f"   Max sigma level: {summary['max_sigma_level']:.1f}")
            
            # Check specific crisis periods
            crisis_periods = [
                ('2007-2008 Food Crisis', '2007-01-01', '2008-12-31'),
                ('2010-2011 Volatility', '2010-06-01', '2011-09-30'),
                ('COVID-19 Impact', '2020-03-01', '2020-12-31'),
                ('Ukraine Crisis', '2022-02-01', '2023-12-31')
            ]
            
            for period_name, start_date, end_date in crisis_periods:
                period_start = pd.to_datetime(start_date)
                period_end = pd.to_datetime(end_date)
                
                # Filter anomalies in this period
                period_mask = (
                    (df_anomalies['date'] >= period_start) & 
                    (df_anomalies['date'] <= period_end) &
                    (df_anomalies['food_price_index_is_anomaly'] == True)
                )
                period_anomalies = df_anomalies[period_mask]
                
                if not period_anomalies.empty:
                    max_sigma = period_anomalies['food_price_index_sigma_level'].max()
                    print(f"   {period_name}: {len(period_anomalies)} anomalies (max {max_sigma:.1f}σ)")
                else:
                    print(f"   {period_name}: No anomalies detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in historical testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_known_spike_dates():
    """Test detection of specific known spike dates"""
    print("\n📈 Testing Known Spike Dates...")
    
    try:
        df, error = load_fao_data()
        if error or df is None:
            print("❌ Could not load data")
            return False
        
        # Known historical spike periods (approximate)
        known_spikes = [
            ('2008-04-01', "Rice Crisis Peak"),
            ('2008-06-01', "Wheat/Corn Crisis"),
            ('2010-08-01', "Russian Export Ban"),
            ('2011-02-01', "Arab Spring Impact"),
            ('2020-04-01', "COVID-19 Initial Impact"),
            ('2022-03-01', "Ukraine War Start"),
            ('2022-05-01', "Ukraine War Peak")
        ]
        
        # Use moderate sensitivity
        df_anomalies = detect_anomalies(df, 'food_price_index', sigma=2.0, window=60)
        
        spike_detection_results = []
        
        for spike_date, description in known_spikes:
            target_date = pd.to_datetime(spike_date)
            
            # Look for anomalies within 3 months of target date
            window_start = target_date - pd.DateOffset(months=1)
            window_end = target_date + pd.DateOffset(months=2)
            
            window_mask = (
                (df_anomalies['date'] >= window_start) & 
                (df_anomalies['date'] <= window_end) &
                (df_anomalies['food_price_index_is_anomaly'] == True)
            )
            
            window_anomalies = df_anomalies[window_mask]
            
            if not window_anomalies.empty:
                # Find the highest sigma anomaly in this window
                max_anomaly = window_anomalies.loc[window_anomalies['food_price_index_sigma_level'].idxmax()]
                detected_date = max_anomaly['date']
                sigma_level = max_anomaly['food_price_index_sigma_level']
                severity = classify_anomaly_severity(sigma_level)
                
                spike_detection_results.append({
                    'Expected': spike_date,
                    'Description': description,
                    'Detected': detected_date.strftime('%Y-%m-%d'),
                    'Sigma Level': f"{sigma_level:.1f}",
                    'Severity': severity,
                    'Status': '✅ Detected'
                })
                
                print(f"   ✅ {description}: {sigma_level:.1f}σ on {detected_date.strftime('%Y-%m-%d')}")
            else:
                spike_detection_results.append({
                    'Expected': spike_date,
                    'Description': description,
                    'Detected': 'Not detected',
                    'Sigma Level': '-',
                    'Severity': '-',
                    'Status': '❌ Missed'
                })
                print(f"   ❌ {description}: No anomaly detected")
        
        # Summary
        detected_count = len([r for r in spike_detection_results if r['Status'] == '✅ Detected'])
        total_count = len(spike_detection_results)
        detection_rate = (detected_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n📊 Detection Summary:")
        print(f"   Detected: {detected_count}/{total_count} ({detection_rate:.0f}%)")
        
        # Create results dataframe for analysis
        results_df = pd.DataFrame(spike_detection_results)
        print(f"\n📋 Detailed Results:")
        for _, row in results_df.iterrows():
            print(f"   {row['Description']}: {row['Status']}")
        
        return detection_rate >= 60  # At least 60% detection rate
        
    except Exception as e:
        print(f"❌ Error in spike testing: {str(e)}")
        return False

def test_anomaly_severity_classification():
    """Test anomaly severity classification"""
    print("\n🎯 Testing Anomaly Severity Classification...")
    
    test_cases = [
        (1.2, "Weak"),
        (1.8, "Mild"),
        (2.3, "Moderate"), 
        (2.7, "Severe"),
        (3.5, "Extreme")
    ]
    
    all_passed = True
    
    for sigma_level, expected_severity in test_cases:
        actual_severity = classify_anomaly_severity(sigma_level)
        
        if actual_severity == expected_severity:
            print(f"   ✅ {sigma_level}σ → {actual_severity}")
        else:
            print(f"   ❌ {sigma_level}σ → Expected: {expected_severity}, Got: {actual_severity}")
            all_passed = False
    
    return all_passed

def test_multiple_indices():
    """Test anomaly detection across multiple food indices"""
    print("\n🌾 Testing Multiple Indices...")
    
    try:
        df, error = load_fao_data()
        if error or df is None:
            print("❌ Could not load data")
            return False
        
        # Test all available indices
        test_indices = ['food_price_index', 'cereals', 'meat', 'dairy', 'oils', 'sugar']
        available_indices = [idx for idx in test_indices if idx in df.columns]
        
        print(f"   Testing indices: {available_indices}")
        
        results = {}
        
        for index in available_indices:
            try:
                df_anomalies = detect_anomalies(df, index, sigma=2.0)
                summary = get_anomaly_summary(df_anomalies, index)
                
                results[index] = {
                    'total_anomalies': summary['total_anomalies'],
                    'anomaly_rate': summary['anomaly_rate'],
                    'max_sigma': summary['max_sigma_level']
                }
                
                print(f"   {index}: {summary['total_anomalies']} anomalies ({summary['anomaly_rate']:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ {index}: Error - {str(e)}")
                results[index] = None
        
        # Check that we got results for most indices
        successful_indices = len([r for r in results.values() if r is not None])
        success_rate = (successful_indices / len(available_indices) * 100) if available_indices else 0
        
        print(f"   Success rate: {successful_indices}/{len(available_indices)} ({success_rate:.0f}%)")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Error in multi-index testing: {str(e)}")
        return False

def main():
    """Run all anomaly detection tests"""
    print("=" * 60)
    print("🧪 ANOMALY DETECTION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Historical Spike Detection", test_historical_spike_detection),
        ("Known Spike Dates", test_known_spike_dates),
        ("Severity Classification", test_anomaly_severity_classification),
        ("Multiple Indices", test_multiple_indices),
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
        print("\n🎉 All tests passed! Anomaly detection is working correctly.")
        print("The system can detect historical food price crises.")
    elif success_rate >= 75:
        print("\n✅ Most tests passed. Anomaly detection is functional.")
        print("Minor issues may exist but core functionality works.")
    else:
        print("\n⚠️ Several tests failed. Review anomaly detection implementation.")
    
    print("\n💡 To test in the dashboard:")
    print("1. Run: streamlit run app.py")
    print("2. Enable 'Anomaly Detection' in sidebar")
    print("3. Adjust sensitivity and observe highlighted anomalies")
    print("4. Look for red/blue markers on the chart")
    print("5. Check the anomaly analysis section below chart")

if __name__ == "__main__":
    main()