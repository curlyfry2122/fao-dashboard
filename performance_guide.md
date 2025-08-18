# Performance Monitoring Guide

## Overview
The FAO Dashboard now includes comprehensive performance monitoring that:
- ⏱️ Times each operation automatically
- 📊 Logs to console in debug mode  
- 🚨 Alerts if any operation > 3 seconds
- 📈 Provides detailed performance analytics
- 🧪 Includes comprehensive benchmarking

## Quick Start

### Enable Debug Mode
```bash
export DEBUG_PERFORMANCE=true
```

### Run Application with Monitoring
```bash
DEBUG_PERFORMANCE=true streamlit run app.py
```

### View Performance Summary
```python
from performance_monitor import print_performance_summary
print_performance_summary()
```

## Performance Monitoring Features

### Automatic Operation Timing
All key operations are automatically monitored:
- **Data Pipeline**: Data fetching, validation, parsing, metrics calculation
- **Chart Generation**: All chart types (Line, Area, Year-over-Year)
- **Analytics**: KPI calculations, correlation analysis
- **Export Operations**: Excel generation, pivot tables
- **Streamlit Operations**: Data loading, UI rendering

### Real-time Console Output
```
⏱️  data_pipeline_run: 0.408s ✅ SUCCESS
⏱️  chart_build: 0.104s ✅ SUCCESS  
⏱️  slow_operation: 4.005s ✅ SUCCESS 🐌 SLOW
🚨 SLOW OPERATION DETECTED: slow_operation took 4.005s (threshold: 3.0s)
```

### Performance Analytics
```python
from performance_monitor import get_performance_summary

# Get detailed performance summary
summary = get_performance_summary(hours=24)
print(f"Total operations: {summary['summary']['total_operations']}")
print(f"Success rate: {summary['summary']['success_rate']:.1%}")
print(f"Slow operations: {summary['summary']['slow_operations']}")
```

## Benchmark Suite

### Run Full Benchmark
```bash
python3 benchmark.py
```

### Sample Output
```
🚀 FAO Dashboard Performance Benchmark Suite
============================================================
📅 Start Time: 2025-08-18 09:58:45
💾 Initial Memory: 146.5 MB

📊 Data Pipeline Benchmarks
----------------------------------------
🔧 Benchmarking: Data Pipeline - Monthly (Full History)
   Time: 0.408s ✅ PASS
   Memory Δ: +0.2 MB
   result_shape: (428, 25)

📈 Chart Generation Benchmarks  
----------------------------------------
🔧 Benchmarking: Chart Generation - Line Chart
   Time: 0.104s ✅ PASS

🔍 Analytics Benchmarks
----------------------------------------
🔧 Benchmarking: KPI Calculation
   Time: 0.045s ✅ PASS

📋 Benchmark Suite Summary
============================================================
⏰ Total Time: 12.3 seconds
💾 Final Memory: 152.1 MB
📈 Memory Change: +5.6 MB
```

## Configuration Options

### Environment Variables
```bash
# Enable/disable debug output
export DEBUG_PERFORMANCE=true

# Skip slow tests in benchmark
export SKIP_SLOW_TESTS=true
```

### Alert Threshold
The default alert threshold is 3 seconds. Operations taking longer will trigger alerts.

### Custom Monitoring
```python
from performance_monitor import performance_monitor, performance_context

# Decorator approach
@performance_monitor('my_operation', include_args=True)
def my_function():
    # Function implementation
    pass

# Context manager approach  
with performance_context('data_processing', {'rows': len(df)}):
    # Process data
    result = complex_operation(df)
```

## Performance Benchmarks

### Expected Performance (Typical Hardware)
- **Data Pipeline (Monthly)**: < 1s (cached), < 5s (fresh)
- **Chart Generation**: < 2s for standard charts
- **KPI Calculation**: < 1s
- **Correlation Analysis**: < 3s
- **Excel Export**: < 3s

### Memory Usage
- **Baseline**: ~100-150 MB
- **With Full Dataset**: ~200-300 MB  
- **Memory Leak Tolerance**: < 50 MB increase over 10 iterations

## Troubleshooting

### Slow Operations
If operations consistently exceed expected times:
1. Check network connectivity (for data fetching)
2. Verify dataset size isn't unexpectedly large
3. Monitor memory usage for potential memory pressure
4. Run benchmark suite to identify bottlenecks

### Memory Issues
```python
# Check memory usage trends
from performance_monitor import PerformanceMonitor
recent_metrics = PerformanceMonitor.get_recent_metrics(hours=1)
for metric in recent_metrics:
    print(f"{metric.operation_name}: {metric.metadata.get('memory_delta_mb', 'N/A')} MB")
```

### Log Files
Performance logs are automatically saved to:
- `.performance_logs/performance_YYYYMMDD.jsonl`
- Logs include detailed timing and metadata
- Logs rotate daily to prevent disk space issues

## Integration Examples

### Streamlit Integration
```python
import streamlit as st
from performance_monitor import get_performance_summary

# Add performance metrics to sidebar
if st.sidebar.button("Show Performance Stats"):
    summary = get_performance_summary(hours=1)
    st.sidebar.json(summary)
```

### Production Monitoring
```python
# Monitor for performance degradation
summary = get_performance_summary(hours=24)
slow_rate = summary['summary']['slow_operation_rate']
if slow_rate > 0.1:  # More than 10% slow operations
    print("⚠️ Performance degradation detected")
```

## Best Practices

1. **Enable debug mode during development**
2. **Run benchmarks before deploying changes**
3. **Monitor slow operation trends in production**
4. **Set up alerts for performance degradation**
5. **Regular performance testing with full historical data**

## Performance Tips

1. **Cache Management**: Ensure proper cache TTL settings
2. **Data Filtering**: Filter large datasets early in pipeline
3. **Chart Optimization**: Limit data points for visualization
4. **Memory Management**: Clear large objects when done
5. **Concurrent Operations**: Be aware of resource contention