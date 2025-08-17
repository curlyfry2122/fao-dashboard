"""
Automation Configuration for FAO Dashboard Weekly Updates

Simple configuration management for the weekly updater and monitoring system.
Provides sensible defaults with easy customization options.
"""

import os
from pathlib import Path
from typing import Optional


class AutomationConfig:
    """
    Configuration manager for FAO dashboard automation.
    
    Provides default settings with optional customization via environment
    variables or configuration file.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with defaults and optional overrides.
        
        Args:
            config_path: Optional path to configuration file (future use)
        """
        # Base paths
        self.project_root = Path(__file__).parent.parent
        self.automation_dir = Path(__file__).parent
        self.logs_dir = self.automation_dir / "logs"
        
        # Cache settings
        self.cache_ttl_hours = float(os.getenv('FAO_CACHE_TTL_HOURS', '168'))  # 1 week default
        
        # Logging settings
        self.log_level = os.getenv('FAO_LOG_LEVEL', 'INFO')
        self.console_logging = os.getenv('FAO_CONSOLE_LOGGING', 'true').lower() == 'true'
        self.log_retention_days = int(os.getenv('FAO_LOG_RETENTION_DAYS', '30'))
        
        # Update scheduling
        self.update_day = os.getenv('FAO_UPDATE_DAY', 'sunday')  # Day of week for updates
        self.update_hour = int(os.getenv('FAO_UPDATE_HOUR', '2'))  # Hour of day (24h format)
        self.update_minute = int(os.getenv('FAO_UPDATE_MINUTE', '0'))  # Minute of hour
        
        # Notification settings
        self.notifications_enabled = os.getenv('FAO_NOTIFICATIONS_ENABLED', 'true').lower() == 'true'
        self.notify_on_success = os.getenv('FAO_NOTIFY_ON_SUCCESS', 'false').lower() == 'true'
        self.notify_on_error = os.getenv('FAO_NOTIFY_ON_ERROR', 'true').lower() == 'true'
        
        # Email notification settings (for future use)
        self.email_enabled = os.getenv('FAO_EMAIL_ENABLED', 'false').lower() == 'true'
        self.email_smtp_server = os.getenv('FAO_EMAIL_SMTP_SERVER', '')
        self.email_smtp_port = int(os.getenv('FAO_EMAIL_SMTP_PORT', '587'))
        self.email_username = os.getenv('FAO_EMAIL_USERNAME', '')
        self.email_password = os.getenv('FAO_EMAIL_PASSWORD', '')
        self.email_from = os.getenv('FAO_EMAIL_FROM', '')
        self.email_to = os.getenv('FAO_EMAIL_TO', '').split(',') if os.getenv('FAO_EMAIL_TO') else []
        
        # Monitoring settings
        self.monitoring_enabled = os.getenv('FAO_MONITORING_ENABLED', 'true').lower() == 'true'
        self.health_check_frequency = os.getenv('FAO_HEALTH_CHECK_FREQUENCY', 'weekly')
        
        # Data validation thresholds
        self.max_data_age_days = int(os.getenv('FAO_MAX_DATA_AGE_DAYS', '60'))
        self.min_expected_rows = int(os.getenv('FAO_MIN_EXPECTED_ROWS', '300'))
        self.max_missing_values_percent = float(os.getenv('FAO_MAX_MISSING_VALUES_PERCENT', '5.0'))
        
        # Create necessary directories
        self.ensure_directories()
        
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.logs_dir.mkdir(exist_ok=True)
        
    def get_cron_schedule(self) -> str:
        """
        Generate cron schedule string based on configuration.
        
        Returns:
            Cron schedule string (e.g., "0 2 * * 0" for Sunday at 2 AM)
        """
        # Map day names to cron day numbers (0 = Sunday)
        day_mapping = {
            'sunday': '0',
            'monday': '1', 
            'tuesday': '2',
            'wednesday': '3',
            'thursday': '4',
            'friday': '5',
            'saturday': '6'
        }
        
        day_num = day_mapping.get(self.update_day.lower(), '0')
        
        return f"{self.update_minute} {self.update_hour} * * {day_num}"
        
    def get_log_file_path(self, log_name: str) -> Path:
        """Get path for a specific log file."""
        return self.logs_dir / f"{log_name}.log"
        
    def get_environment_summary(self) -> dict:
        """Get summary of current configuration for debugging."""
        return {
            'project_root': str(self.project_root),
            'logs_dir': str(self.logs_dir),
            'cache_ttl_hours': self.cache_ttl_hours,
            'log_level': self.log_level,
            'console_logging': self.console_logging,
            'update_schedule': self.get_cron_schedule(),
            'notifications_enabled': self.notifications_enabled,
            'monitoring_enabled': self.monitoring_enabled,
            'email_enabled': self.email_enabled
        }
        
    def validate_configuration(self) -> list:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of configuration issues (empty if all valid)
        """
        issues = []
        
        # Validate update schedule
        if self.update_hour < 0 or self.update_hour > 23:
            issues.append(f"Invalid update_hour: {self.update_hour} (must be 0-23)")
            
        if self.update_minute < 0 or self.update_minute > 59:
            issues.append(f"Invalid update_minute: {self.update_minute} (must be 0-59)")
            
        # Validate day of week
        valid_days = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
        if self.update_day.lower() not in valid_days:
            issues.append(f"Invalid update_day: {self.update_day} (must be one of {valid_days})")
            
        # Validate cache TTL
        if self.cache_ttl_hours <= 0:
            issues.append(f"Invalid cache_ttl_hours: {self.cache_ttl_hours} (must be positive)")
            
        # Validate thresholds
        if self.max_data_age_days <= 0:
            issues.append(f"Invalid max_data_age_days: {self.max_data_age_days} (must be positive)")
            
        if self.min_expected_rows <= 0:
            issues.append(f"Invalid min_expected_rows: {self.min_expected_rows} (must be positive)")
            
        if self.max_missing_values_percent < 0 or self.max_missing_values_percent > 100:
            issues.append(f"Invalid max_missing_values_percent: {self.max_missing_values_percent} (must be 0-100)")
            
        # Validate email settings if email is enabled
        if self.email_enabled:
            if not self.email_smtp_server:
                issues.append("Email enabled but no SMTP server configured")
            if not self.email_from:
                issues.append("Email enabled but no sender address configured")
            if not self.email_to:
                issues.append("Email enabled but no recipient addresses configured")
                
        return issues


# Create a default configuration instance for easy import
default_config = AutomationConfig()


def print_configuration_help():
    """Print help about configuration options."""
    help_text = """
FAO Dashboard Automation Configuration

Environment Variables:
  FAO_CACHE_TTL_HOURS=168           Cache time-to-live in hours (default: 1 week)
  FAO_LOG_LEVEL=INFO                Logging level (DEBUG, INFO, WARNING, ERROR)
  FAO_CONSOLE_LOGGING=true          Enable console logging (true/false)
  FAO_LOG_RETENTION_DAYS=30         Days to keep log files
  
  FAO_UPDATE_DAY=sunday             Day of week for updates
  FAO_UPDATE_HOUR=2                 Hour of day for updates (0-23)
  FAO_UPDATE_MINUTE=0               Minute of hour for updates (0-59)
  
  FAO_NOTIFICATIONS_ENABLED=true    Enable notifications (true/false)
  FAO_NOTIFY_ON_SUCCESS=false       Notify on successful updates
  FAO_NOTIFY_ON_ERROR=true          Notify on errors
  
  FAO_EMAIL_ENABLED=false           Enable email notifications
  FAO_EMAIL_SMTP_SERVER=            SMTP server for email
  FAO_EMAIL_SMTP_PORT=587           SMTP port
  FAO_EMAIL_USERNAME=               SMTP username
  FAO_EMAIL_PASSWORD=               SMTP password
  FAO_EMAIL_FROM=                   Sender email address
  FAO_EMAIL_TO=                     Recipient emails (comma-separated)
  
  FAO_MONITORING_ENABLED=true       Enable monitoring features
  FAO_HEALTH_CHECK_FREQUENCY=weekly Health check frequency
  
  FAO_MAX_DATA_AGE_DAYS=60          Alert if data older than this
  FAO_MIN_EXPECTED_ROWS=300         Alert if fewer rows than this
  FAO_MAX_MISSING_VALUES_PERCENT=5  Alert if more missing values than this

Examples:
  # Run updates on Monday at 3:30 AM
  export FAO_UPDATE_DAY=monday
  export FAO_UPDATE_HOUR=3
  export FAO_UPDATE_MINUTE=30
  
  # Enable email notifications
  export FAO_EMAIL_ENABLED=true
  export FAO_EMAIL_SMTP_SERVER=smtp.gmail.com
  export FAO_EMAIL_FROM=dashboard@yourcompany.com
  export FAO_EMAIL_TO=admin@yourcompany.com,team@yourcompany.com
"""
    print(help_text)


if __name__ == "__main__":
    # Print configuration help when run directly
    print_configuration_help()
    
    # Validate and display current configuration
    config = AutomationConfig()
    issues = config.validate_configuration()
    
    print("\nCurrent Configuration:")
    for key, value in config.get_environment_summary().items():
        print(f"  {key}: {value}")
        
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ Configuration is valid")