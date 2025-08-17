#!/usr/bin/env python3
"""
Weekly FAO Data Monitor

Integrates with the existing fao-data-monitor agent to provide automated
health checks and monitoring reports for the FAO dashboard.

Usage:
    python weekly_monitor.py                    # Run weekly health check
    python weekly_monitor.py --quick            # Quick status check
    python weekly_monitor.py --report           # Generate detailed report
    python weekly_monitor.py --history          # Show monitoring history
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from automation_config import AutomationConfig
from data_pipeline import DataPipeline


class WeeklyMonitor:
    """
    Weekly monitoring service for FAO Food Price Index data.
    
    Provides health checks, data validation, and monitoring reports
    with integration to the existing fao-data-monitor agent.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the weekly monitor with configuration."""
        self.config = AutomationConfig(config_path)
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Configure logging for the monitor."""
        # Create logs directory if it doesn't exist
        Path(self.config.logs_dir).mkdir(exist_ok=True)
        
        # Configure logging
        log_file = Path(self.config.logs_dir) / "weekly_monitor.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) if self.config.console_logging else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def check_cache_health(self) -> Dict[str, Dict]:
        """Check health of data caches."""
        results = {}
        
        for sheet_name in ['Monthly', 'Annual']:
            try:
                pipeline = DataPipeline(sheet_name=sheet_name)
                status = pipeline.get_cache_status()
                
                health_info = {
                    'exists': status['exists'],
                    'is_valid': status['is_valid'],
                    'age_hours': status['age'].total_seconds() / 3600 if status['age'] else None,
                    'ttl_remaining_hours': status['ttl_remaining'].total_seconds() / 3600 if status['ttl_remaining'] else None,
                    'status': 'OK'
                }
                
                # Determine health status
                if not status['exists']:
                    health_info['status'] = 'ERROR'
                    health_info['issue'] = 'Cache does not exist'
                elif not status['is_valid']:
                    health_info['status'] = 'WARNING'
                    health_info['issue'] = 'Cache is expired'
                elif health_info['age_hours'] and health_info['age_hours'] > 168:  # > 1 week
                    health_info['status'] = 'WARNING'
                    health_info['issue'] = 'Cache is older than expected'
                    
                results[sheet_name] = health_info
                
            except Exception as e:
                results[sheet_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                
        return results
        
    def check_data_freshness(self) -> Dict[str, Dict]:
        """Check freshness of data in caches."""
        results = {}
        
        for sheet_name in ['Monthly', 'Annual']:
            try:
                pipeline = DataPipeline(sheet_name=sheet_name)
                
                if not pipeline.get_cache_status()['exists']:
                    results[sheet_name] = {
                        'status': 'ERROR',
                        'issue': 'No cached data available'
                    }
                    continue
                    
                df = pipeline._load_from_cache()
                
                if df is None or len(df) == 0:
                    results[sheet_name] = {
                        'status': 'ERROR',
                        'issue': 'Cache contains no data'
                    }
                    continue
                    
                # Check data freshness
                latest_date = df['date'].max()
                data_age_days = (datetime.now() - latest_date).days
                
                freshness_info = {
                    'latest_date': latest_date.strftime('%Y-%m-%d'),
                    'age_days': data_age_days,
                    'row_count': len(df),
                    'status': 'OK'
                }
                
                # Determine freshness status
                if data_age_days > self.config.max_data_age_days:
                    freshness_info['status'] = 'WARNING'
                    freshness_info['issue'] = f'Data is {data_age_days} days old (max: {self.config.max_data_age_days})'
                elif len(df) < self.config.min_expected_rows:
                    freshness_info['status'] = 'WARNING'
                    freshness_info['issue'] = f'Only {len(df)} rows (min expected: {self.config.min_expected_rows})'
                    
                # Check for missing values
                missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                freshness_info['missing_values_percent'] = round(missing_percent, 2)
                
                if missing_percent > self.config.max_missing_values_percent:
                    freshness_info['status'] = 'WARNING'
                    if 'issue' not in freshness_info:
                        freshness_info['issue'] = f'{missing_percent:.1f}% missing values (max: {self.config.max_missing_values_percent}%)'
                        
                results[sheet_name] = freshness_info
                
            except Exception as e:
                results[sheet_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                
        return results
        
    def run_quick_check(self) -> Dict:
        """Run quick health check."""
        self.logger.info("Running quick health check")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'quick',
            'cache_health': self.check_cache_health(),
            'overall_status': 'OK'
        }
        
        # Determine overall status
        for sheet_info in result['cache_health'].values():
            if sheet_info.get('status') == 'ERROR':
                result['overall_status'] = 'ERROR'
                break
            elif sheet_info.get('status') == 'WARNING':
                result['overall_status'] = 'WARNING'
                
        return result
        
    def run_comprehensive_check(self) -> Dict:
        """Run comprehensive health check including data freshness."""
        self.logger.info("Running comprehensive health check")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'comprehensive',
            'cache_health': self.check_cache_health(),
            'data_freshness': self.check_data_freshness(),
            'overall_status': 'OK',
            'summary': {}
        }
        
        # Determine overall status
        all_checks = [result['cache_health'], result['data_freshness']]
        
        error_count = 0
        warning_count = 0
        
        for check_group in all_checks:
            for sheet_info in check_group.values():
                if sheet_info.get('status') == 'ERROR':
                    error_count += 1
                elif sheet_info.get('status') == 'WARNING':
                    warning_count += 1
                    
        if error_count > 0:
            result['overall_status'] = 'ERROR'
        elif warning_count > 0:
            result['overall_status'] = 'WARNING'
            
        # Add summary
        result['summary'] = {
            'total_checks': len(result['cache_health']) + len(result['data_freshness']),
            'errors': error_count,
            'warnings': warning_count,
            'ok_count': result['summary'].get('total_checks', 0) - error_count - warning_count
        }
        
        return result
        
    def save_monitoring_report(self, report: Dict) -> Path:
        """Save monitoring report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = Path(self.config.logs_dir) / f"monitor_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Monitoring report saved to {report_file}")
            return report_file
        except Exception as e:
            self.logger.error(f"Failed to save monitoring report: {e}")
            raise
            
    def get_monitoring_history(self, days: int = 30) -> List[Dict]:
        """Get recent monitoring history."""
        history = []
        
        # Find all monitoring report files
        report_files = list(Path(self.config.logs_dir).glob("monitor_report_*.json"))
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for report_file in sorted(report_files, reverse=True):
            try:
                # Extract timestamp from filename
                timestamp_str = report_file.stem.split('_', 2)[-1]  # Get timestamp part
                report_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if report_date >= cutoff_date:
                    with open(report_file) as f:
                        report_data = json.load(f)
                    history.append(report_data)
                    
            except (ValueError, json.JSONDecodeError) as e:
                self.logger.warning(f"Could not parse report file {report_file}: {e}")
                continue
                
        return history
        
    def print_report(self, report: Dict) -> None:
        """Print monitoring report in human-readable format."""
        print("=" * 60)
        print(f"FAO Dashboard Monitoring Report - {report['timestamp']}")
        print("=" * 60)
        
        print(f"\n📊 Overall Status: {self._get_status_emoji(report['overall_status'])} {report['overall_status']}")
        
        if 'summary' in report:
            summary = report['summary']
            print(f"\n📈 Summary:")
            print(f"   Total Checks: {summary['total_checks']}")
            print(f"   ✅ OK: {summary['ok_count']}")
            print(f"   ⚠️  Warnings: {summary['warnings']}")
            print(f"   ❌ Errors: {summary['errors']}")
            
        # Cache Health
        print(f"\n💾 Cache Health:")
        for sheet_name, health in report['cache_health'].items():
            status_emoji = self._get_status_emoji(health.get('status', 'UNKNOWN'))
            print(f"   {status_emoji} {sheet_name}:")
            
            if 'error' in health:
                print(f"      Error: {health['error']}")
            else:
                print(f"      Exists: {health.get('exists', 'Unknown')}")
                print(f"      Valid: {health.get('is_valid', 'Unknown')}")
                if health.get('age_hours') is not None:
                    print(f"      Age: {health['age_hours']:.1f} hours")
                if 'issue' in health:
                    print(f"      Issue: {health['issue']}")
                    
        # Data Freshness (if available)
        if 'data_freshness' in report:
            print(f"\n🔄 Data Freshness:")
            for sheet_name, freshness in report['data_freshness'].items():
                status_emoji = self._get_status_emoji(freshness.get('status', 'UNKNOWN'))
                print(f"   {status_emoji} {sheet_name}:")
                
                if 'error' in freshness:
                    print(f"      Error: {freshness['error']}")
                else:
                    print(f"      Latest Date: {freshness.get('latest_date', 'Unknown')}")
                    print(f"      Age: {freshness.get('age_days', 'Unknown')} days")
                    print(f"      Rows: {freshness.get('row_count', 'Unknown')}")
                    print(f"      Missing Values: {freshness.get('missing_values_percent', 'Unknown')}%")
                    if 'issue' in freshness:
                        print(f"      Issue: {freshness['issue']}")
                        
        print("=" * 60)
        
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        emoji_map = {
            'OK': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'UNKNOWN': '❓'
        }
        return emoji_map.get(status, '❓')
        
    def print_history_summary(self, history: List[Dict]) -> None:
        """Print summary of monitoring history."""
        if not history:
            print("No monitoring history found")
            return
            
        print("=" * 60)
        print(f"Monitoring History Summary ({len(history)} reports)")
        print("=" * 60)
        
        # Count statuses
        status_counts = {'OK': 0, 'WARNING': 0, 'ERROR': 0}
        
        print(f"\n📅 Recent Reports:")
        for report in history[:10]:  # Show last 10
            timestamp = report.get('timestamp', 'Unknown')
            status = report.get('overall_status', 'UNKNOWN')
            emoji = self._get_status_emoji(status)
            
            # Parse timestamp for display
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                time_str = timestamp
                
            print(f"   {emoji} {time_str} - {status}")
            
            if status in status_counts:
                status_counts[status] += 1
                
        # Overall statistics
        print(f"\n📊 Statistics:")
        total = len(history)
        for status, count in status_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            emoji = self._get_status_emoji(status)
            print(f"   {emoji} {status}: {count}/{total} ({percentage:.1f}%)")
            
        print("=" * 60)


def main():
    """Main entry point for the weekly monitor."""
    parser = argparse.ArgumentParser(description="Weekly FAO Data Monitor")
    parser.add_argument('--quick', action='store_true', help='Run quick status check only')
    parser.add_argument('--report', action='store_true', help='Generate and save detailed report')
    parser.add_argument('--history', action='store_true', help='Show monitoring history')
    parser.add_argument('--days', type=int, default=30, help='Days of history to show (default: 30)')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        monitor = WeeklyMonitor(args.config)
        
        if args.history:
            # Show monitoring history
            history = monitor.get_monitoring_history(args.days)
            monitor.print_history_summary(history)
            return
            
        if args.quick:
            # Quick check only
            report = monitor.run_quick_check()
        else:
            # Comprehensive check
            report = monitor.run_comprehensive_check()
            
        # Print report
        monitor.print_report(report)
        
        # Save report if requested
        if args.report:
            report_file = monitor.save_monitoring_report(report)
            print(f"\n📁 Report saved to: {report_file}")
            
        # Exit with appropriate code
        if report['overall_status'] == 'ERROR':
            sys.exit(1)
        elif report['overall_status'] == 'WARNING':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()