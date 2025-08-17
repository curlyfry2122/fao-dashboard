# FAO Dashboard Weekly Automation

Simple weekly automation system for the FAO Food Price Index Dashboard that ensures data stays current without manual intervention.

## 🚀 Quick Start

1. **Install the automation:**
   ```bash
   ./setup_weekly_automation.sh install
   ```

2. **Check status:**
   ```bash
   ./setup_weekly_automation.sh status
   ```

3. **Test the system:**
   ```bash
   ./setup_weekly_automation.sh test
   ```

## 📋 What It Does

- **Weekly Data Updates**: Automatically refreshes FAO data cache every Sunday at 2 AM
- **Health Monitoring**: Checks data freshness and system health
- **Intelligent Caching**: Only updates when needed, preserves historical data
- **Error Recovery**: Graceful fallback if updates fail
- **Logging**: Comprehensive logs for troubleshooting

## 📁 Components

- `weekly_updater.py` - Main update script
- `weekly_monitor.py` - Health monitoring and reporting
- `automation_config.py` - Configuration management
- `setup_weekly_automation.sh` - Installation and setup script
- `logs/` - Log files and monitoring reports

## ⚙️ Configuration

The system uses environment variables for configuration:

```bash
# Update schedule (default: Sunday at 2:00 AM)
export FAO_UPDATE_DAY=sunday
export FAO_UPDATE_HOUR=2
export FAO_UPDATE_MINUTE=0

# Cache settings (default: 1 week)
export FAO_CACHE_TTL_HOURS=168

# Notifications (default: log only)
export FAO_NOTIFICATIONS_ENABLED=true
export FAO_NOTIFY_ON_SUCCESS=false
export FAO_NOTIFY_ON_ERROR=true
```

For complete configuration options:
```bash
python3 automation_config.py
```

## 🔧 Manual Operations

### Force Update
```bash
python3 weekly_updater.py --force
```

### Check Cache Status
```bash
python3 weekly_updater.py --check
```

### Health Monitoring
```bash
python3 weekly_monitor.py --report
python3 weekly_monitor.py --history
```

### View Logs
```bash
tail -f automation/logs/weekly_updater.log
tail -f automation/logs/weekly_monitor.log
```

## 🛠 Troubleshooting

### Common Issues

**No cron job installed:**
```bash
./setup_weekly_automation.sh install
```

**Updates not running:**
```bash
./setup_weekly_automation.sh status
python3 weekly_updater.py --check
```

**Data appears outdated:**
```bash
python3 weekly_updater.py --force
python3 weekly_monitor.py --report
```

### Log Files

- `automation/logs/weekly_updater.log` - Update process logs
- `automation/logs/weekly_monitor.log` - Monitoring logs
- `automation/logs/cron.log` - Cron job execution logs
- `automation/logs/notifications.log` - System notifications

## 🔒 Safety Features

- **Lock Files**: Prevents concurrent updates
- **Historical Preservation**: Never modifies existing cached data
- **Graceful Fallback**: Uses cached data if fresh fetch fails
- **Validation**: Checks data integrity before caching
- **Resource Limits**: Minimal CPU and bandwidth usage

## 📊 Monitoring

The system generates monitoring reports showing:

- Cache health and age
- Data freshness and quality
- Missing values and anomalies
- System performance metrics
- Historical trends

Example monitoring output:
```
📊 Overall Status: ✅ OK

💾 Cache Health:
   ✅ Monthly: Age=2.1 hours, Valid=True
   ✅ Annual: Age=2.1 hours, Valid=True

🔄 Data Freshness:
   ✅ Monthly: Latest=2025-07-01, Rows=428
   ✅ Annual: Latest=2025-01-01, Rows=36
```

## 🚫 Uninstall

To remove the automation:
```bash
./setup_weekly_automation.sh uninstall
```

## 📞 Support

For issues or questions:
1. Check the logs in `automation/logs/`
2. Run the test command: `./setup_weekly_automation.sh test`
3. Check configuration: `python3 automation_config.py`
4. Review the dashboard at http://localhost:8503

---

**Default Schedule**: Every Sunday at 2:00 AM  
**Resource Usage**: ~5 minutes weekly, minimal CPU/bandwidth  
**Maintenance**: None required under normal operation