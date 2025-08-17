#!/usr/bin/env python3
"""
Weekly FAO Data Updater

Simple background service that refreshes FAO Food Price Index data caches
on a weekly schedule. Designed for minimal resource usage and easy maintenance.

Usage:
    python weekly_updater.py              # Run update
    python weekly_updater.py --check      # Check status only
    python weekly_updater.py --force      # Force update regardless of cache age
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import DataPipeline
from automation_config import AutomationConfig


class WeeklyUpdater:
    """
    Weekly background updater for FAO Food Price Index data.
    
    Handles cache refresh, validation, and logging with minimal resource usage.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the weekly updater with configuration."""
        self.config = AutomationConfig(config_path)
        self.setup_logging()
        self.lock_file = Path(self.config.logs_dir) / "weekly_updater.lock"
        
    def setup_logging(self) -> None:
        """Configure logging for the updater."""
        # Create logs directory if it doesn't exist
        Path(self.config.logs_dir).mkdir(exist_ok=True)
        
        # Configure logging
        log_file = Path(self.config.logs_dir) / "weekly_updater.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) if self.config.console_logging else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def is_locked(self) -> bool:
        """Check if another update process is running."""
        if not self.lock_file.exists():
            return False
            
        try:
            # Check if lock is stale (older than 1 hour)
            lock_age = time.time() - self.lock_file.stat().st_mtime
            if lock_age > 3600:  # 1 hour
                self.logger.warning("Removing stale lock file")
                self.lock_file.unlink()
                return False
            return True
        except Exception:
            return False
            
    def acquire_lock(self) -> bool:
        """Acquire update lock to prevent concurrent runs."""
        if self.is_locked():
            self.logger.error("Another update process is already running")
            return False
            
        try:
            self.lock_file.write_text(str(os.getpid()))
            return True
        except Exception as e:
            self.logger.error(f"Failed to acquire lock: {e}")
            return False
            
    def release_lock(self) -> None:
        """Release update lock."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            self.logger.error(f"Failed to release lock: {e}")
            
    def should_update(self, sheet_name: str, force: bool = False) -> bool:
        """Determine if cache should be updated."""
        if force:
            return True
            
        try:
            pipeline = DataPipeline(sheet_name=sheet_name, cache_ttl_hours=self.config.cache_ttl_hours)
            status = pipeline.get_cache_status()
            
            if not status['exists']:
                self.logger.info(f"No cache exists for {sheet_name}, update needed")
                return True
                
            # Check if cache is older than weekly threshold
            cache_age_hours = status['age'].total_seconds() / 3600 if status['age'] else float('inf')
            weekly_threshold = 7 * 24  # 7 days in hours
            
            if cache_age_hours > weekly_threshold:
                self.logger.info(f"Cache for {sheet_name} is {cache_age_hours:.1f} hours old, update needed")
                return True
                
            self.logger.info(f"Cache for {sheet_name} is {cache_age_hours:.1f} hours old, no update needed")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking cache status for {sheet_name}: {e}")
            return True  # Update on error to be safe
            
    def update_cache(self, sheet_name: str) -> Tuple[bool, str]:
        """Update cache for specified sheet."""
        try:
            self.logger.info(f"Starting cache update for {sheet_name}")
            start_time = datetime.now()
            
            # Create pipeline with longer cache TTL for background updates
            pipeline = DataPipeline(
                sheet_name=sheet_name, 
                cache_ttl_hours=self.config.cache_ttl_hours
            )
            
            # Run the pipeline (this will fetch and cache new data)
            df = pipeline.run()
            
            # Validate results
            if df is None or len(df) == 0:
                return False, "No data returned from pipeline"
                
            # Check data freshness
            latest_date = df['date'].max()
            data_age_days = (datetime.now() - latest_date).days
            
            if data_age_days > 60:  # More than 60 days old
                self.logger.warning(f"Data appears outdated: latest date is {latest_date}")
                
            duration = datetime.now() - start_time
            self.logger.info(f"Successfully updated {sheet_name} cache in {duration.total_seconds():.1f} seconds")
            self.logger.info(f"Data range: {df['date'].min()} to {df['date'].max()}, {len(df)} rows")
            
            return True, f"Updated {len(df)} rows, latest: {latest_date.strftime('%Y-%m-%d')}"
            
        except Exception as e:
            self.logger.error(f"Failed to update cache for {sheet_name}: {e}")
            return False, str(e)
            
    def run_update(self, force: bool = False) -> Dict[str, Dict]:
        """Run the weekly update process."""
        self.logger.info("=" * 60)
        self.logger.info("Starting weekly FAO data update")
        self.logger.info(f"Force update: {force}")
        
        results = {}
        
        for sheet_name in ['Monthly', 'Annual']:
            result = {
                'attempted': False,
                'success': False,
                'message': '',
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                if self.should_update(sheet_name, force):
                    result['attempted'] = True
                    success, message = self.update_cache(sheet_name)
                    result['success'] = success
                    result['message'] = message
                else:
                    result['message'] = 'Cache is current, no update needed'
                    result['success'] = True  # Not updating is also success
                    
            except Exception as e:
                result['attempted'] = True
                result['success'] = False
                result['message'] = f"Update failed: {e}"
                self.logger.error(f"Error updating {sheet_name}: {e}")
                
            results[sheet_name] = result
            
        # Send notification if configured and there were issues
        self.send_notification_if_needed(results)
        
        self.logger.info("Weekly update completed")
        self.logger.info("=" * 60)
        
        return results
        
    def send_notification_if_needed(self, results: Dict[str, Dict]) -> None:
        """Send notification if there were issues and notifications are enabled."""
        if not self.config.notifications_enabled:
            return
            
        issues = []
        successes = []
        
        for sheet_name, result in results.items():
            if result['attempted'] and not result['success']:
                issues.append(f"{sheet_name}: {result['message']}")
            elif result['success']:
                successes.append(f"{sheet_name}: {result['message']}")
                
        if issues:
            self.send_notification("Weekly FAO Update - Issues Found", issues, successes)
        elif self.config.notify_on_success and successes:
            self.send_notification("Weekly FAO Update - Success", [], successes)
            
    def send_notification(self, subject: str, issues: list, successes: list) -> None:
        """Send notification about update results."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            message_parts = [
                f"FAO Dashboard Weekly Update Report",
                f"Timestamp: {timestamp}",
                ""
            ]
            
            if successes:
                message_parts.extend([
                    "✅ Successful Updates:",
                    *[f"  • {success}" for success in successes],
                    ""
                ])
                
            if issues:
                message_parts.extend([
                    "❌ Issues Found:",
                    *[f"  • {issue}" for issue in issues],
                    ""
                ])
                
            message_parts.extend([
                "Dashboard URL: http://localhost:8503",
                "Log file: " + str(Path(self.config.logs_dir) / "weekly_updater.log")
            ])
            
            message = "\n".join(message_parts)
            
            # For now, just log the notification
            # In the future, could add email/Slack integration
            self.logger.info(f"NOTIFICATION: {subject}")
            self.logger.info(message)
            
            # Write notification to separate file for external processing
            notification_file = Path(self.config.logs_dir) / "notifications.log"
            with open(notification_file, 'a') as f:
                f.write(f"\n{timestamp} - {subject}\n")
                f.write(message + "\n")
                f.write("-" * 40 + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            
    def check_status(self) -> Dict[str, Dict]:
        """Check current cache status without updating."""
        status = {}
        
        for sheet_name in ['Monthly', 'Annual']:
            try:
                pipeline = DataPipeline(sheet_name=sheet_name)
                cache_status = pipeline.get_cache_status()
                
                result = {
                    'exists': cache_status['exists'],
                    'age_hours': cache_status['age'].total_seconds() / 3600 if cache_status['age'] else None,
                    'is_valid': cache_status['is_valid'],
                    'needs_update': False
                }
                
                if result['age_hours']:
                    result['needs_update'] = result['age_hours'] > (7 * 24)  # Older than 1 week
                    
                status[sheet_name] = result
                
            except Exception as e:
                status[sheet_name] = {'error': str(e)}
                
        return status


def main():
    """Main entry point for the weekly updater."""
    parser = argparse.ArgumentParser(description="Weekly FAO Data Updater")
    parser.add_argument('--check', action='store_true', help='Check status only, do not update')
    parser.add_argument('--force', action='store_true', help='Force update regardless of cache age')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        updater = WeeklyUpdater(args.config)
        
        if args.check:
            # Check status only
            status = updater.check_status()
            print("Cache Status:")
            for sheet_name, info in status.items():
                if 'error' in info:
                    print(f"  {sheet_name}: ERROR - {info['error']}")
                else:
                    age_str = f"{info['age_hours']:.1f}h" if info['age_hours'] else "N/A"
                    update_needed = "YES" if info['needs_update'] else "NO"
                    print(f"  {sheet_name}: Age={age_str}, Valid={info['is_valid']}, Update Needed={update_needed}")
            return
            
        # Acquire lock for update
        if not updater.acquire_lock():
            print("Another update process is already running")
            sys.exit(1)
            
        try:
            # Run update
            results = updater.run_update(force=args.force)
            
            # Print summary
            print("Update Summary:")
            for sheet_name, result in results.items():
                status = "SUCCESS" if result['success'] else "FAILED"
                print(f"  {sheet_name}: {status} - {result['message']}")
                
        finally:
            updater.release_lock()
            
    except KeyboardInterrupt:
        print("\nUpdate interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()