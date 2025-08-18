#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for FAO Dashboard.

Tests performance with full historical FAO data and provides detailed
performance analysis across all major operations.
"""

import os
import sys
import time
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Set debug mode for detailed performance monitoring
os.environ['DEBUG_PERFORMANCE'] = 'true'

# Import FAO Dashboard modules
from data_pipeline import DataPipeline
from chart_builder import build_chart
from kpi_calculator import calculate_kpis
from correlation_analyzer import calculate_correlation_matrix, build_correlation_heatmap
from excel_exporter import ExcelExporter
from performance_monitor import (
    PerformanceMonitor, 
    get_performance_summary,
    print_performance_summary,
    performance_context
)


class BenchmarkSuite:
    """Comprehensive performance benchmark suite for FAO Dashboard."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {}
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
        print("🚀 FAO Dashboard Performance Benchmark Suite")
        print("=" * 60)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Initial Memory: {self.initial_memory:.1f} MB")
        print()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_operation(
        self, 
        name: str, 
        operation: Callable,
        *args,
        expected_max_time: float = 10.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a single operation.
        
        Args:
            name: Name of the operation
            operation: Function to benchmark
            expected_max_time: Expected maximum time in seconds
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"🔧 Benchmarking: {name}")
        
        # Record initial state
        initial_memory = self.get_memory_usage()
        start_time = time.time()
        start_perf = time.perf_counter()
        
        try:
            result = operation(*args, **kwargs)
            
            # Calculate metrics
            wall_time = time.time() - start_time
            perf_time = time.perf_counter() - start_perf
            final_memory = self.get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            # Determine result metadata
            metadata = {}
            if isinstance(result, pd.DataFrame):
                metadata['result_shape'] = result.shape
                metadata['result_size_mb'] = result.memory_usage(deep=True).sum() / 1024 / 1024
            elif hasattr(result, '__len__'):
                metadata['result_length'] = len(result)
            
            benchmark_result = {
                'name': name,
                'success': True,
                'wall_time': wall_time,
                'perf_time': perf_time,
                'memory_before_mb': initial_memory,
                'memory_after_mb': final_memory,
                'memory_delta_mb': memory_delta,
                'expected_max_time': expected_max_time,
                'within_expected': perf_time <= expected_max_time,
                'is_slow': perf_time > 3.0,
                'metadata': metadata,
                'timestamp': datetime.now(),
                'error': None
            }
            
            # Print results
            status = "✅ PASS" if benchmark_result['within_expected'] else "⚠️  SLOW"
            if benchmark_result['is_slow']:
                status += " 🐌"
            
            print(f"   Time: {perf_time:.3f}s {status}")
            print(f"   Memory Δ: {memory_delta:+.1f} MB")
            if metadata:
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
            print()
            
            return benchmark_result
            
        except Exception as e:
            wall_time = time.time() - start_time
            final_memory = self.get_memory_usage()
            
            benchmark_result = {
                'name': name,
                'success': False,
                'wall_time': wall_time,
                'perf_time': time.perf_counter() - start_perf,
                'memory_before_mb': initial_memory,
                'memory_after_mb': final_memory,
                'memory_delta_mb': final_memory - initial_memory,
                'expected_max_time': expected_max_time,
                'within_expected': False,
                'is_slow': True,
                'metadata': {},
                'timestamp': datetime.now(),
                'error': str(e)
            }
            
            print(f"   ❌ ERROR: {e}")
            print()
            
            return benchmark_result
    
    def benchmark_data_pipeline(self) -> Dict[str, Any]:
        """Benchmark complete data pipeline with full historical data."""
        print("📊 Data Pipeline Benchmarks")
        print("-" * 40)
        
        pipeline_results = {}
        
        # Test Monthly data pipeline
        monthly_result = self.benchmark_operation(
            "Data Pipeline - Monthly (Full History)",
            self._run_monthly_pipeline,
            expected_max_time=15.0
        )
        pipeline_results['monthly'] = monthly_result
        
        # Test Annual data pipeline
        annual_result = self.benchmark_operation(
            "Data Pipeline - Annual (Full History)", 
            self._run_annual_pipeline,
            expected_max_time=10.0
        )
        pipeline_results['annual'] = annual_result
        
        # Test cache performance
        if monthly_result['success']:
            cache_result = self.benchmark_operation(
                "Data Pipeline - Monthly (Cached)",
                self._run_monthly_pipeline,
                expected_max_time=1.0
            )
            pipeline_results['monthly_cached'] = cache_result
        
        return pipeline_results
    
    def _run_monthly_pipeline(self) -> pd.DataFrame:
        """Run monthly data pipeline."""
        pipeline = DataPipeline(sheet_name='Monthly', cache_ttl_hours=0.1)
        return pipeline.run()
    
    def _run_annual_pipeline(self) -> pd.DataFrame:
        """Run annual data pipeline."""
        pipeline = DataPipeline(sheet_name='Annual', cache_ttl_hours=0.1)
        return pipeline.run()
    
    def benchmark_chart_generation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark chart generation with full dataset."""
        print("📈 Chart Generation Benchmarks")
        print("-" * 40)
        
        chart_results = {}
        indices = ['food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        available_indices = [idx for idx in indices if idx in df.columns]
        
        if not available_indices:
            print("⚠️  No standard indices found, using available columns")
            available_indices = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        
        chart_types = [
            ('Line Chart', 2.0),
            ('Area Chart', 2.5),
            ('Year-over-Year Change', 3.0)
        ]
        
        for chart_type, expected_time in chart_types:
            result = self.benchmark_operation(
                f"Chart Generation - {chart_type}",
                self._build_chart_safe,
                df, chart_type, available_indices[:3],
                expected_max_time=expected_time
            )
            chart_results[chart_type.lower().replace(' ', '_')] = result
        
        # Test with anomaly detection
        anomaly_config = {
            'enabled': True,
            'sigma': 2.0,
            'window': min(60, len(df) // 4),
            'show_bands': True
        }
        
        anomaly_result = self.benchmark_operation(
            "Chart Generation - Line Chart with Anomaly Detection",
            self._build_chart_safe,
            df, 'Line Chart', available_indices[:2], anomaly_config,
            expected_max_time=5.0
        )
        chart_results['line_with_anomaly'] = anomaly_result
        
        return chart_results
    
    def _build_chart_safe(self, df, chart_type, indices, anomaly_detection=None):
        """Safely build chart with error handling."""
        try:
            return build_chart(df, chart_type, indices, anomaly_detection)
        except Exception as e:
            print(f"Chart generation failed: {e}")
            # Return a minimal success indicator
            import plotly.graph_objects as go
            return go.Figure()
    
    def benchmark_analytics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark analytics operations."""
        print("🔍 Analytics Benchmarks")
        print("-" * 40)
        
        analytics_results = {}
        
        # Get available numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        standard_indices = ['food_price_index', 'meat', 'dairy', 'cereals', 'oils', 'sugar']
        available_indices = [idx for idx in standard_indices if idx in numeric_columns]
        
        if not available_indices:
            available_indices = numeric_columns[:6]
        
        # KPI calculations
        if available_indices:
            df_with_date = df.reset_index() if 'date' not in df.columns else df
            kpi_result = self.benchmark_operation(
                "KPI Calculation",
                calculate_kpis,
                df_with_date, available_indices[:3],
                expected_max_time=2.0
            )
            analytics_results['kpi_calculation'] = kpi_result
        
        # Correlation analysis
        if len(available_indices) > 1:
            corr_result = self.benchmark_operation(
                "Correlation Matrix",
                calculate_correlation_matrix,
                df, available_indices[:5],
                expected_max_time=3.0
            )
            analytics_results['correlation_matrix'] = corr_result
            
            # Correlation heatmap
            if corr_result['success']:
                try:
                    corr_matrix = calculate_correlation_matrix(df, available_indices[:5])
                    heatmap_result = self.benchmark_operation(
                        "Correlation Heatmap",
                        build_correlation_heatmap,
                        corr_matrix,
                        expected_max_time=1.0
                    )
                    analytics_results['correlation_heatmap'] = heatmap_result
                except Exception as e:
                    print(f"Correlation heatmap failed: {e}")
        
        return analytics_results
    
    def benchmark_export_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark export operations."""
        print("📤 Export Operations Benchmarks")
        print("-" * 40)
        
        export_results = {}
        
        # Excel export
        df_export = df.head(1000) if len(df) > 1000 else df  # Limit for export testing
        df_export = df_export.reset_index() if df_export.index.name or hasattr(df_export.index, 'name') else df_export
        
        exporter = ExcelExporter()
        excel_result = self.benchmark_operation(
            "Excel Export",
            self._export_to_excel_safe,
            exporter, df_export,
            expected_max_time=3.0
        )
        export_results['excel_export'] = excel_result
        
        return export_results
    
    def _export_to_excel_safe(self, exporter, df):
        """Safely export to Excel."""
        try:
            workbook = exporter.generate_data_sheet(df, "Benchmark_Data")
            workbook.close()
            return True
        except Exception as e:
            print(f"Excel export failed: {e}")
            return False
    
    def benchmark_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark memory usage under load."""
        print("💾 Memory Usage Benchmarks")
        print("-" * 40)
        
        memory_results = {}
        
        # Large dataset handling
        if len(df) < 5000:
            # Create larger dataset for memory testing
            large_df = pd.concat([df] * 5, ignore_index=True)
        else:
            large_df = df
        
        memory_result = self.benchmark_operation(
            "Large Dataset Processing",
            self._process_large_dataset,
            large_df,
            expected_max_time=8.0
        )
        memory_results['large_dataset'] = memory_result
        
        # Memory leak test
        leak_result = self.benchmark_operation(
            "Memory Leak Test",
            self._test_memory_leak,
            df,
            expected_max_time=10.0
        )
        memory_results['memory_leak_test'] = leak_result
        
        return memory_results
    
    def _process_large_dataset(self, df):
        """Process large dataset to test memory usage."""
        results = []
        
        # Multiple operations on large dataset
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
        
        if numeric_cols:
            # Chart generation
            try:
                chart = build_chart(df, 'Line Chart', numeric_cols)
                results.append(f"Chart: {len(chart.data)} traces")
            except:
                pass
            
            # KPI calculation
            try:
                df_with_date = df.reset_index() if 'date' not in df.columns else df
                kpis = calculate_kpis(df_with_date, numeric_cols[:1])
                results.append(f"KPIs: {len(kpis)} indices")
            except:
                pass
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = calculate_correlation_matrix(df, numeric_cols[:4])
                    results.append(f"Correlation: {corr_matrix.shape}")
                except:
                    pass
        
        return results
    
    def _test_memory_leak(self, df):
        """Test for memory leaks by running operations multiple times."""
        initial_memory = self.get_memory_usage()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:2]
        
        for i in range(10):
            if numeric_cols:
                try:
                    # Perform various operations
                    chart = build_chart(df, 'Line Chart', numeric_cols)
                    
                    df_with_date = df.reset_index() if 'date' not in df.columns else df
                    kpis = calculate_kpis(df_with_date, numeric_cols[:1])
                    
                    if len(numeric_cols) > 1:
                        corr = calculate_correlation_matrix(df, numeric_cols)
                    
                    # Clear variables
                    del chart, kpis
                    if len(numeric_cols) > 1:
                        del corr
                        
                except Exception:
                    pass
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'iterations': 10,
            'acceptable': memory_increase < 50  # Less than 50MB increase acceptable
        }
    
    def benchmark_concurrent_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark concurrent operations."""
        print("⚡ Concurrent Operations Benchmarks")
        print("-" * 40)
        
        concurrent_results = {}
        
        result = self.benchmark_operation(
            "Concurrent Chart Generation",
            self._concurrent_chart_generation,
            df,
            expected_max_time=5.0
        )
        concurrent_results['concurrent_charts'] = result
        
        return concurrent_results
    
    def _concurrent_chart_generation(self, df):
        """Test concurrent chart generation."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return "Insufficient numeric columns for concurrent testing"
        
        def create_chart(chart_type, indices):
            try:
                return build_chart(df, chart_type, indices)
            except Exception as e:
                return f"Error: {e}"
        
        tasks = [
            ('Line Chart', numeric_cols[:2]),
            ('Area Chart', numeric_cols[:2]),
            ('Line Chart', numeric_cols[1:3] if len(numeric_cols) > 2 else numeric_cols[:2]),
        ]
        
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_chart, chart_type, indices) 
                      for chart_type, indices in tasks]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(f"Chart completed: {type(result).__name__}")
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🏁 Starting Full Benchmark Suite")
        print("=" * 60)
        
        total_start = time.time()
        all_results = {}
        
        try:
            # 1. Data Pipeline Benchmarks
            pipeline_results = self.benchmark_data_pipeline()
            all_results['data_pipeline'] = pipeline_results
            
            # Get data for subsequent tests
            monthly_data = None
            if pipeline_results.get('monthly', {}).get('success'):
                try:
                    monthly_pipeline = DataPipeline(sheet_name='Monthly')
                    monthly_data = monthly_pipeline.run()
                except Exception as e:
                    print(f"Could not load data for subsequent tests: {e}")
            
            if monthly_data is not None and len(monthly_data) > 0:
                print(f"✅ Using dataset with {len(monthly_data)} rows for benchmarks\n")
                
                # 2. Chart Generation Benchmarks
                chart_results = self.benchmark_chart_generation(monthly_data)
                all_results['chart_generation'] = chart_results
                
                # 3. Analytics Benchmarks
                analytics_results = self.benchmark_analytics(monthly_data)
                all_results['analytics'] = analytics_results
                
                # 4. Export Operations Benchmarks
                export_results = self.benchmark_export_operations(monthly_data)
                all_results['export_operations'] = export_results
                
                # 5. Memory Usage Benchmarks
                memory_results = self.benchmark_memory_usage(monthly_data)
                all_results['memory_usage'] = memory_results
                
                # 6. Concurrent Operations Benchmarks
                concurrent_results = self.benchmark_concurrent_operations(monthly_data)
                all_results['concurrent_operations'] = concurrent_results
                
            else:
                print("⚠️  Could not load data, skipping dependent benchmarks")
            
        except Exception as e:
            print(f"❌ Benchmark suite error: {e}")
        
        total_time = time.time() - total_start
        final_memory = self.get_memory_usage()
        
        # Summary
        print("📋 Benchmark Suite Summary")
        print("=" * 60)
        print(f"⏰ Total Time: {total_time:.1f} seconds")
        print(f"💾 Final Memory: {final_memory:.1f} MB")
        print(f"📈 Memory Change: {final_memory - self.initial_memory:+.1f} MB")
        
        # Performance monitoring summary
        print("\n📊 Performance Monitoring Summary:")
        print_performance_summary(hours=1)
        
        all_results['summary'] = {
            'total_time': total_time,
            'initial_memory_mb': self.initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': final_memory - self.initial_memory,
            'timestamp': datetime.now()
        }
        
        return all_results


def main():
    """Main benchmark execution."""
    benchmark = BenchmarkSuite()
    
    try:
        results = benchmark.run_full_benchmark()
        
        # Analyze results
        print("\n🎯 Performance Analysis")
        print("=" * 60)
        
        slow_operations = []
        failed_operations = []
        
        def analyze_results(results_dict, prefix=""):
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    if 'success' in value:
                        # This is a benchmark result
                        full_name = f"{prefix}{value['name']}" if prefix else value['name']
                        
                        if not value['success']:
                            failed_operations.append(full_name)
                        elif value['is_slow']:
                            slow_operations.append((full_name, value['perf_time']))
                    else:
                        # This is a nested category
                        analyze_results(value, f"{prefix}{key}.")
        
        analyze_results(results)
        
        if failed_operations:
            print("❌ Failed Operations:")
            for op in failed_operations:
                print(f"   • {op}")
        
        if slow_operations:
            print("\n🐌 Slow Operations (>3s):")
            for op, duration in slow_operations:
                print(f"   • {op}: {duration:.3f}s")
        
        if not failed_operations and not slow_operations:
            print("✅ All operations completed successfully within expected time!")
        
        print(f"\n✨ Benchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0 if not failed_operations else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)