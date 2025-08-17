#!/bin/bash

# FAO Dashboard Weekly Automation Setup Script
#
# This script sets up automated weekly updates for the FAO Food Price Index Dashboard.
# It creates cron jobs, log directories, and validates the setup.
#
# Usage:
#   ./setup_weekly_automation.sh install    # Install automation
#   ./setup_weekly_automation.sh uninstall  # Remove automation  
#   ./setup_weekly_automation.sh status     # Check current status
#   ./setup_weekly_automation.sh test       # Test the update system

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UPDATER_SCRIPT="$SCRIPT_DIR/weekly_updater.py"
MONITOR_SCRIPT="$SCRIPT_DIR/weekly_monitor.py"
LOGS_DIR="$SCRIPT_DIR/logs"
PYTHON_CMD="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if script is run with correct permissions
check_permissions() {
    if [[ ! -w "$HOME" ]]; then
        log_error "Cannot write to home directory. Please check permissions."
        exit 1
    fi
}

# Validate Python environment
validate_python() {
    log_info "Validating Python environment..."
    
    if ! command -v $PYTHON_CMD &> /dev/null; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi
    
    # Check Python version
    python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    log_info "Found Python $python_version"
    
    # Test if required modules are available
    if ! $PYTHON_CMD -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from data_pipeline import DataPipeline" 2>/dev/null; then
        log_error "Cannot import required modules. Please ensure the dashboard dependencies are installed."
        log_info "Try running: pip3 install -r $PROJECT_ROOT/requirements.txt"
        exit 1
    fi
    
    log_success "Python environment validated"
}

# Get cron schedule from configuration
get_cron_schedule() {
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from automation_config import AutomationConfig
config = AutomationConfig()
print(config.get_cron_schedule())
"
}

# Create necessary directories
setup_directories() {
    log_info "Creating directories..."
    
    mkdir -p "$LOGS_DIR"
    
    # Set appropriate permissions
    chmod 755 "$SCRIPT_DIR"
    chmod 755 "$LOGS_DIR"
    chmod +x "$UPDATER_SCRIPT"
    
    log_success "Directories created and permissions set"
}

# Install cron job for weekly updates
install_cron_job() {
    log_info "Installing cron job..."
    
    # Get current crontab
    current_crontab=$(crontab -l 2>/dev/null || echo "")
    
    # Check if our job already exists
    if echo "$current_crontab" | grep -q "weekly_updater.py"; then
        log_warning "Cron job for weekly updates already exists"
        return 0
    fi
    
    # Get schedule from configuration
    cron_schedule=$(get_cron_schedule)
    
    # Create new crontab entry
    cron_entry="$cron_schedule cd $PROJECT_ROOT && $PYTHON_CMD $UPDATER_SCRIPT >> $LOGS_DIR/cron.log 2>&1"
    
    # Add to crontab
    {
        echo "$current_crontab"
        echo "# FAO Dashboard Weekly Update"
        echo "$cron_entry"
    } | crontab -
    
    log_success "Cron job installed: $cron_schedule"
    log_info "Updates will run: $(get_schedule_description)"
}

# Get human-readable schedule description
get_schedule_description() {
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from automation_config import AutomationConfig
config = AutomationConfig()
day = config.update_day.title()
hour = config.update_hour
minute = config.update_minute
time_str = f'{hour:02d}:{minute:02d}'
print(f'Every {day} at {time_str}')
"
}

# Remove cron job
uninstall_cron_job() {
    log_info "Removing cron job..."
    
    # Get current crontab and remove our entries
    current_crontab=$(crontab -l 2>/dev/null || echo "")
    
    if ! echo "$current_crontab" | grep -q "weekly_updater.py"; then
        log_warning "No cron job found for weekly updates"
        return 0
    fi
    
    # Remove our cron job (and the comment line before it)
    new_crontab=$(echo "$current_crontab" | grep -v -E "(weekly_updater.py|FAO Dashboard Weekly Update)")
    
    # Update crontab
    if [[ -n "$new_crontab" ]]; then
        echo "$new_crontab" | crontab -
    else
        crontab -r 2>/dev/null || true
    fi
    
    log_success "Cron job removed"
}

# Setup log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."
    
    # Create logrotate configuration
    logrotate_config="/tmp/fao-dashboard-logs"
    
    cat > "$logrotate_config" << EOF
$LOGS_DIR/*.log {
    weekly
    rotate 8
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}
EOF
    
    # Test logrotate configuration
    if logrotate -t "$logrotate_config" 2>/dev/null; then
        log_success "Log rotation configured (manual rotation only)"
        log_info "To manually rotate logs: logrotate -f $logrotate_config"
    else
        log_warning "Could not configure automatic log rotation"
        log_info "Please manually clean up logs in $LOGS_DIR periodically"
    fi
    
    rm -f "$logrotate_config"
}

# Test the update system
test_update_system() {
    log_info "Testing update system..."
    
    # Test configuration
    log_info "Testing configuration..."
    if ! $PYTHON_CMD "$SCRIPT_DIR/automation_config.py" > /dev/null; then
        log_error "Configuration validation failed"
        return 1
    fi
    
    # Test cache status check
    log_info "Testing cache status check..."
    if ! $PYTHON_CMD "$UPDATER_SCRIPT" --check > /dev/null; then
        log_error "Cache status check failed"
        return 1
    fi
    
    # Test dry run of update (check only, no actual update)
    log_info "Testing update logic..."
    if $PYTHON_CMD "$UPDATER_SCRIPT" --check; then
        log_success "Update system test passed"
        return 0
    else
        log_error "Update system test failed"
        return 1
    fi
}

# Show current status
show_status() {
    echo "======================================"
    echo "FAO Dashboard Automation Status"
    echo "======================================"
    
    # Check if cron job exists
    if crontab -l 2>/dev/null | grep -q "weekly_updater.py"; then
        cron_schedule=$(crontab -l | grep "weekly_updater.py" | head -1 | cut -d' ' -f1-5)
        log_success "✅ Cron job installed: $cron_schedule"
        log_info "   Schedule: $(get_schedule_description)"
    else
        log_warning "❌ No cron job found"
    fi
    
    # Check directories
    if [[ -d "$LOGS_DIR" ]]; then
        log_success "✅ Logs directory exists: $LOGS_DIR"
        log_count=$(find "$LOGS_DIR" -name "*.log" 2>/dev/null | wc -l)
        log_info "   Log files: $log_count"
    else
        log_warning "❌ Logs directory missing"
    fi
    
    # Check Python environment
    if $PYTHON_CMD -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from data_pipeline import DataPipeline" 2>/dev/null; then
        log_success "✅ Python environment OK"
    else
        log_error "❌ Python environment issues"
    fi
    
    # Check cache status
    echo ""
    log_info "Cache Status:"
    $PYTHON_CMD "$UPDATER_SCRIPT" --check 2>/dev/null || log_warning "Could not check cache status"
    
    # Recent log entries
    echo ""
    log_info "Recent log entries:"
    if [[ -f "$LOGS_DIR/weekly_updater.log" ]]; then
        tail -5 "$LOGS_DIR/weekly_updater.log" 2>/dev/null | sed 's/^/   /' || log_info "   No recent log entries"
    else
        log_info "   No log file found"
    fi
}

# Main installation function
install_automation() {
    echo "======================================"
    echo "Installing FAO Dashboard Automation"
    echo "======================================"
    
    check_permissions
    validate_python
    setup_directories
    install_cron_job
    setup_log_rotation
    
    echo ""
    log_success "✅ Installation completed successfully!"
    echo ""
    log_info "Next steps:"
    log_info "1. Test the system: ./setup_weekly_automation.sh test"
    log_info "2. Check status: ./setup_weekly_automation.sh status"
    log_info "3. View logs: tail -f $LOGS_DIR/weekly_updater.log"
    log_info "4. Manual update: $PYTHON_CMD $UPDATER_SCRIPT"
    echo ""
    log_info "Automation will run: $(get_schedule_description)"
}

# Main uninstallation function
uninstall_automation() {
    echo "======================================"
    echo "Uninstalling FAO Dashboard Automation"
    echo "======================================"
    
    uninstall_cron_job
    
    # Optionally remove logs directory
    read -p "Remove logs directory ($LOGS_DIR)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$LOGS_DIR"
        log_success "Logs directory removed"
    else
        log_info "Logs directory preserved"
    fi
    
    log_success "✅ Uninstallation completed"
}

# Main script logic
main() {
    case "${1:-}" in
        install)
            install_automation
            ;;
        uninstall)
            uninstall_automation
            ;;
        status)
            show_status
            ;;
        test)
            test_update_system
            ;;
        *)
            echo "FAO Dashboard Weekly Automation Setup"
            echo ""
            echo "Usage: $0 {install|uninstall|status|test}"
            echo ""
            echo "Commands:"
            echo "  install    - Install weekly automation"
            echo "  uninstall  - Remove weekly automation"
            echo "  status     - Show current status"
            echo "  test       - Test the update system"
            echo ""
            echo "Examples:"
            echo "  $0 install     # Set up weekly updates"
            echo "  $0 status      # Check if automation is working"
            echo "  $0 test        # Test update functionality"
            echo ""
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"