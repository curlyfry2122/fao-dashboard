"""
Performance Monitoring System for FAO Dashboard.

Provides decorators and utilities for monitoring operation performance,
logging debug information, and alerting on slow operations.
"""

import os
import time
import logging
import functools
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
from pathlib import Path
import json

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration
DEBUG_PERFORMANCE = os.getenv('DEBUG_PERFORMANCE', 'false').lower() in ('true', '1', 'yes')
PERFORMANCE_LOG_DIR = Path('.performance_logs')
ALERT_THRESHOLD_SECONDS = 3.0
MAX_LOG_ENTRIES = 1000

# Performance metrics storage
_performance_metrics = []
_metrics_lock = threading.Lock()

# Create log directory
PERFORMANCE_LOG_DIR.mkdir(exist_ok=True)


class PerformanceMetric:
    """Container for performance metric data."""
    
    def __init__(
        self,
        operation_name: str,
        start_time: datetime,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operation_name = operation_name
        self.start_time = start_time
        self.duration = duration
        self.success = success
        self.error = error
        self.metadata = metadata or {}
        self.is_slow = duration > ALERT_THRESHOLD_SECONDS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'duration': self.duration,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata,
            'is_slow': self.is_slow
        }


class PerformanceMonitor:
    """Main performance monitoring class."""
    
    @staticmethod
    def log_metric(metric: PerformanceMetric) -> None:
        """Log a performance metric."""
        with _metrics_lock:
            _performance_metrics.append(metric)
            
            # Keep only recent metrics to prevent memory growth
            if len(_performance_metrics) > MAX_LOG_ENTRIES:
                _performance_metrics[:] = _performance_metrics[-MAX_LOG_ENTRIES:]
        
        # Debug logging
        if DEBUG_PERFORMANCE:
            status = "✅ SUCCESS" if metric.success else "❌ ERROR"
            slow_indicator = "🐌 SLOW" if metric.is_slow else ""
            
            print(f"⏱️  {metric.operation_name}: {metric.duration:.3f}s {status} {slow_indicator}")
            
            if metric.error:
                print(f"   Error: {metric.error}")
            
            if metric.metadata:
                for key, value in metric.metadata.items():
                    print(f"   {key}: {value}")
        
        # Alert on slow operations
        if metric.is_slow:
            PerformanceMonitor.alert_slow_operation(metric)
        
        # Save to log file
        PerformanceMonitor.save_to_log(metric)
    
    @staticmethod
    def alert_slow_operation(metric: PerformanceMetric) -> None:
        """Alert on slow operations."""
        alert_msg = (
            f"🚨 SLOW OPERATION DETECTED: {metric.operation_name} "
            f"took {metric.duration:.3f}s (threshold: {ALERT_THRESHOLD_SECONDS}s)"
        )
        
        if DEBUG_PERFORMANCE:
            print(alert_msg)
        
        logger.warning(alert_msg)
        
        # Could extend this to send notifications, emails, etc.
    
    @staticmethod
    def save_to_log(metric: PerformanceMetric) -> None:
        """Save metric to log file."""
        log_file = PERFORMANCE_LOG_DIR / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                json.dump(metric.to_dict(), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write performance log: {e}")
    
    @staticmethod
    def get_recent_metrics(hours: int = 24) -> List[PerformanceMetric]:
        """Get recent performance metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with _metrics_lock:
            return [
                metric for metric in _performance_metrics
                if metric.start_time >= cutoff_time
            ]
    
    @staticmethod
    def get_slow_operations(hours: int = 24) -> List[PerformanceMetric]:
        """Get recent slow operations."""
        recent_metrics = PerformanceMonitor.get_recent_metrics(hours)
        return [metric for metric in recent_metrics if metric.is_slow]
    
    @staticmethod
    def get_operation_stats(operation_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        recent_metrics = PerformanceMonitor.get_recent_metrics(hours)
        operation_metrics = [m for m in recent_metrics if m.operation_name == operation_name]
        
        if not operation_metrics:
            return {}
        
        durations = [m.duration for m in operation_metrics]
        success_count = sum(1 for m in operation_metrics if m.success)
        slow_count = sum(1 for m in operation_metrics if m.is_slow)
        
        return {
            'operation_name': operation_name,
            'total_calls': len(operation_metrics),
            'success_count': success_count,
            'error_count': len(operation_metrics) - success_count,
            'slow_count': slow_count,
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'success_rate': success_count / len(operation_metrics),
            'slow_rate': slow_count / len(operation_metrics)
        }


def performance_monitor(
    operation_name: Optional[str] = None,
    include_args: bool = False,
    alert_threshold: Optional[float] = None
) -> Callable:
    """
    Decorator for monitoring function performance.
    
    Args:
        operation_name: Custom name for the operation. Defaults to function name.
        include_args: Whether to include function arguments in metadata.
        alert_threshold: Custom alert threshold in seconds.
        
    Example:
        @performance_monitor('data_loading')
        def load_data():
            # Function implementation
            pass
        
        @performance_monitor(include_args=True, alert_threshold=5.0)
        def process_data(df, columns):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = datetime.now()
            metadata = {}
            
            if include_args:
                metadata['args_count'] = len(args)
                metadata['kwargs_keys'] = list(kwargs.keys())
                
                # Include specific metadata for common operations
                if 'df' in kwargs and hasattr(kwargs['df'], 'shape'):
                    metadata['dataframe_shape'] = kwargs['df'].shape
                elif args and hasattr(args[0], 'shape'):
                    metadata['dataframe_shape'] = args[0].shape
            
            try:
                start_perf = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_perf
                
                metric = PerformanceMetric(
                    operation_name=op_name,
                    start_time=start_time,
                    duration=duration,
                    success=True,
                    metadata=metadata
                )
                
                PerformanceMonitor.log_metric(metric)
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start_perf
                
                metric = PerformanceMetric(
                    operation_name=op_name,
                    start_time=start_time,
                    duration=duration,
                    success=False,
                    error=str(e),
                    metadata=metadata
                )
                
                PerformanceMonitor.log_metric(metric)
                raise
        
        return wrapper
    return decorator


@contextmanager
def performance_context(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for monitoring performance of code blocks.
    
    Example:
        with performance_context('data_processing', {'rows': len(df)}):
            # Process data
            processed_df = complex_operation(df)
    """
    start_time = datetime.now()
    start_perf = time.perf_counter()
    
    try:
        yield
        duration = time.perf_counter() - start_perf
        
        metric = PerformanceMetric(
            operation_name=operation_name,
            start_time=start_time,
            duration=duration,
            success=True,
            metadata=metadata or {}
        )
        
        PerformanceMonitor.log_metric(metric)
        
    except Exception as e:
        duration = time.perf_counter() - start_perf
        
        metric = PerformanceMetric(
            operation_name=operation_name,
            start_time=start_time,
            duration=duration,
            success=False,
            error=str(e),
            metadata=metadata or {}
        )
        
        PerformanceMonitor.log_metric(metric)
        raise


def get_performance_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a comprehensive performance summary.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        Dictionary with performance statistics
    """
    recent_metrics = PerformanceMonitor.get_recent_metrics(hours)
    slow_operations = PerformanceMonitor.get_slow_operations(hours)
    
    if not recent_metrics:
        return {'message': 'No performance data available'}
    
    # Overall statistics
    total_operations = len(recent_metrics)
    successful_operations = sum(1 for m in recent_metrics if m.success)
    slow_operations_count = len(slow_operations)
    
    durations = [m.duration for m in recent_metrics]
    avg_duration = sum(durations) / len(durations)
    
    # Group by operation name
    operation_groups = {}
    for metric in recent_metrics:
        op_name = metric.operation_name
        if op_name not in operation_groups:
            operation_groups[op_name] = []
        operation_groups[op_name].append(metric)
    
    # Calculate stats for each operation
    operation_stats = {}
    for op_name, metrics in operation_groups.items():
        op_durations = [m.duration for m in metrics]
        operation_stats[op_name] = {
            'calls': len(metrics),
            'avg_duration': sum(op_durations) / len(op_durations),
            'max_duration': max(op_durations),
            'slow_calls': sum(1 for m in metrics if m.is_slow),
            'success_rate': sum(1 for m in metrics if m.success) / len(metrics)
        }
    
    return {
        'summary': {
            'time_period_hours': hours,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'failed_operations': total_operations - successful_operations,
            'slow_operations': slow_operations_count,
            'success_rate': successful_operations / total_operations,
            'slow_operation_rate': slow_operations_count / total_operations,
            'average_duration': avg_duration,
            'max_duration': max(durations),
            'min_duration': min(durations)
        },
        'operation_breakdown': operation_stats,
        'recent_slow_operations': [
            {
                'operation': op.operation_name,
                'duration': op.duration,
                'time': op.start_time.isoformat(),
                'error': op.error
            }
            for op in slow_operations[-10:]  # Last 10 slow operations
        ]
    }


def print_performance_summary(hours: int = 24) -> None:
    """Print a formatted performance summary to console."""
    summary = get_performance_summary(hours)
    
    if 'message' in summary:
        print(summary['message'])
        return
    
    print("📊 Performance Summary")
    print("=" * 50)
    
    stats = summary['summary']
    print(f"⏰ Time Period: {hours} hours")
    print(f"📈 Total Operations: {stats['total_operations']}")
    print(f"✅ Success Rate: {stats['success_rate']:.1%}")
    print(f"🐌 Slow Operations: {stats['slow_operations']} ({stats['slow_operation_rate']:.1%})")
    print(f"⏱️  Average Duration: {stats['average_duration']:.3f}s")
    print(f"🔥 Max Duration: {stats['max_duration']:.3f}s")
    
    print("\n📋 Operation Breakdown:")
    for op_name, op_stats in summary['operation_breakdown'].items():
        print(f"  {op_name}:")
        print(f"    Calls: {op_stats['calls']}")
        print(f"    Avg Duration: {op_stats['avg_duration']:.3f}s")
        print(f"    Max Duration: {op_stats['max_duration']:.3f}s")
        print(f"    Success Rate: {op_stats['success_rate']:.1%}")
        if op_stats['slow_calls'] > 0:
            print(f"    🐌 Slow Calls: {op_stats['slow_calls']}")
    
    if summary['recent_slow_operations']:
        print("\n🚨 Recent Slow Operations:")
        for slow_op in summary['recent_slow_operations']:
            print(f"  {slow_op['operation']}: {slow_op['duration']:.3f}s at {slow_op['time']}")
            if slow_op['error']:
                print(f"    Error: {slow_op['error']}")