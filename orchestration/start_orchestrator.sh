#!/bin/bash

# FAO Dashboard Agent Orchestrator Startup Script
# This script provides a comprehensive way to start, stop, and manage the orchestration system

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLI_SCRIPT="$SCRIPT_DIR/orchestrator_cli.py"
PID_FILE="$SCRIPT_DIR/orchestrator.pid"
LOG_FILE="$PROJECT_DIR/automation/logs/orchestrator_startup.log"
CONFIG_FILE="$SCRIPT_DIR/orchestrator_config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}❌ ERROR: $1${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}✅ $1${NC}"
}

# Warning message
warning() {
    log "${YELLOW}⚠️ $1${NC}"
}

# Info message
info() {
    log "${BLUE}ℹ️ $1${NC}"
}

# Check if orchestrator is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 is required but not installed"
    fi
    
    # Check required packages
    local required_packages=("json" "yaml" "pandas" "requests")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            warning "Python package '$package' not found. Installing..."
            pip3 install "$package" || error_exit "Failed to install $package"
        fi
    done
    
    # Check configuration files
    local config_files=("$CONFIG_FILE" "$SCRIPT_DIR/matrix_config.yml" "$SCRIPT_DIR/escalation_config.json")
    for config in "${config_files[@]}"; do
        if [ ! -f "$config" ]; then
            error_exit "Configuration file not found: $config"
        fi
    done
    
    # Check CLI script
    if [ ! -f "$CLI_SCRIPT" ]; then
        error_exit "CLI script not found: $CLI_SCRIPT"
    fi
    
    if [ ! -x "$CLI_SCRIPT" ]; then
        chmod +x "$CLI_SCRIPT"
        info "Made CLI script executable"
    fi
    
    success "Prerequisites check completed"
}

# Validate configuration
validate_config() {
    info "Validating configuration files..."
    
    # Validate JSON configs
    local json_configs=("$CONFIG_FILE" "$SCRIPT_DIR/escalation_config.json")
    for config in "${json_configs[@]}"; do
        if ! python3 -m json.tool "$config" > /dev/null; then
            error_exit "Invalid JSON configuration: $config"
        fi
    done
    
    # Validate YAML config
    if ! python3 -c "import yaml; yaml.safe_load(open('$SCRIPT_DIR/matrix_config.yml'))" 2>/dev/null; then
        error_exit "Invalid YAML configuration: $SCRIPT_DIR/matrix_config.yml"
    fi
    
    success "Configuration validation completed"
}

# Setup logging directory
setup_logging() {
    local log_dir="$PROJECT_DIR/automation/logs"
    mkdir -p "$log_dir"
    touch "$LOG_FILE"
    info "Logging initialized: $LOG_FILE"
}

# Start the orchestrator
start_orchestrator() {
    if is_running; then
        warning "Orchestrator is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    info "Starting FAO Dashboard Agent Orchestrator..."
    
    # Start orchestrator in background
    nohup python3 "$CLI_SCRIPT" start > "$PROJECT_DIR/automation/logs/orchestrator.log" 2>&1 &
    local pid=$!
    
    # Save PID
    echo "$pid" > "$PID_FILE"
    
    # Wait a moment and check if it's still running
    sleep 3
    if ps -p "$pid" > /dev/null 2>&1; then
        success "Orchestrator started successfully (PID: $pid)"
        info "Logs: $PROJECT_DIR/automation/logs/orchestrator.log"
        info "Use 'orchestration/start_orchestrator.sh status' to check status"
    else
        rm -f "$PID_FILE"
        error_exit "Failed to start orchestrator. Check logs for details."
    fi
}

# Stop the orchestrator
stop_orchestrator() {
    if ! is_running; then
        warning "Orchestrator is not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    info "Stopping orchestrator (PID: $pid)..."
    
    # Send SIGTERM first
    if kill -TERM "$pid" 2>/dev/null; then
        # Wait up to 30 seconds for graceful shutdown
        local count=0
        while [ $count -lt 30 ] && ps -p "$pid" > /dev/null 2>&1; do
            sleep 1
            ((count++))
        done
        
        # If still running, force kill
        if ps -p "$pid" > /dev/null 2>&1; then
            warning "Graceful shutdown timed out, forcing stop..."
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    
    rm -f "$PID_FILE"
    success "Orchestrator stopped"
}

# Restart the orchestrator
restart_orchestrator() {
    info "Restarting orchestrator..."
    stop_orchestrator
    sleep 2
    start_orchestrator
}

# Show orchestrator status
show_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        success "Orchestrator is running (PID: $pid)"
        
        # Show basic system info
        info "System Information:"
        echo "  Process: $(ps -p "$pid" -o pid,ppid,cmd --no-headers)"
        echo "  Uptime: $(ps -p "$pid" -o etime --no-headers | tr -d ' ')"
        echo "  Log file: $PROJECT_DIR/automation/logs/orchestrator.log"
        
        # Try to get status from CLI
        if command -v timeout &> /dev/null; then
            info "Getting orchestrator status..."
            timeout 10s python3 "$CLI_SCRIPT" status 2>/dev/null || warning "Could not retrieve detailed status"
        fi
    else
        warning "Orchestrator is not running"
        return 1
    fi
}

# Monitor the orchestrator
monitor_orchestrator() {
    info "Starting orchestrator monitoring (Press Ctrl+C to stop)..."
    
    if ! is_running; then
        error_exit "Orchestrator is not running. Start it first with: $0 start"
    fi
    
    # Start monitoring
    python3 "$CLI_SCRIPT" monitor --interval 10
}

# Show logs
show_logs() {
    local log_files=(
        "$PROJECT_DIR/automation/logs/orchestrator.log"
        "$PROJECT_DIR/automation/logs/orchestrator_startup.log"
        "$PROJECT_DIR/automation/logs/escalations.log"
    )
    
    info "Available log files:"
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            echo "  📄 $log_file"
            if [ "${1:-}" = "--tail" ]; then
                echo "  Last 10 lines:"
                tail -n 10 "$log_file" | sed 's/^/    /'
                echo
            fi
        else
            echo "  ❌ $log_file (not found)"
        fi
    done
    
    if [ "${1:-}" != "--tail" ]; then
        info "Use '$0 logs --tail' to see recent log entries"
    fi
}

# Health check
health_check() {
    info "Performing orchestrator health check..."
    
    local health_score=0
    local max_score=5
    
    # Check if running
    if is_running; then
        success "Process is running"
        ((health_score++))
    else
        warning "Process is not running"
    fi
    
    # Check configuration files
    if validate_config &>/dev/null; then
        success "Configuration files are valid"
        ((health_score++))
    else
        warning "Configuration validation failed"
    fi
    
    # Check log files
    if [ -f "$PROJECT_DIR/automation/logs/orchestrator.log" ]; then
        success "Log files exist"
        ((health_score++))
    else
        warning "Log files missing"
    fi
    
    # Check dependencies
    if python3 -c "import json, yaml, pandas, requests" &>/dev/null; then
        success "Python dependencies available"
        ((health_score++))
    else
        warning "Some Python dependencies missing"
    fi
    
    # Check CLI functionality
    if python3 "$CLI_SCRIPT" --help &>/dev/null; then
        success "CLI interface functional"
        ((health_score++))
    else
        warning "CLI interface issues"
    fi
    
    # Calculate health percentage
    local health_percentage=$((health_score * 100 / max_score))
    
    info "Health Score: $health_score/$max_score ($health_percentage%)"
    
    if [ $health_percentage -ge 80 ]; then
        success "System health is GOOD"
        return 0
    elif [ $health_percentage -ge 60 ]; then
        warning "System health is FAIR - some issues detected"
        return 1
    else
        warning "System health is POOR - multiple issues detected"
        return 2
    fi
}

# Generate report
generate_report() {
    local report_format="${1:-html}"
    info "Generating $report_format report..."
    
    if ! is_running; then
        warning "Orchestrator is not running. Starting temporarily for report generation..."
        start_orchestrator
        sleep 5
    fi
    
    python3 "$CLI_SCRIPT" report --format "$report_format" --open-report
}

# Install as system service (Linux/macOS)
install_service() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux systemd service
        local service_file="/etc/systemd/system/fao-orchestrator.service"
        info "Installing systemd service..."
        
        sudo tee "$service_file" > /dev/null <<EOF
[Unit]
Description=FAO Dashboard Agent Orchestrator
After=network.target

[Service]
Type=forking
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$SCRIPT_DIR/start_orchestrator.sh start
ExecStop=$SCRIPT_DIR/start_orchestrator.sh stop
ExecReload=$SCRIPT_DIR/start_orchestrator.sh restart
PIDFile=$PID_FILE
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable fao-orchestrator
        success "Systemd service installed. Use 'sudo systemctl start fao-orchestrator' to start"
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS launchd service
        local plist_file="$HOME/Library/LaunchAgents/com.fao.orchestrator.plist"
        info "Installing launchd service..."
        
        mkdir -p "$(dirname "$plist_file")"
        tee "$plist_file" > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.fao.orchestrator</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCRIPT_DIR/start_orchestrator.sh</string>
        <string>start</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF
        
        launchctl load "$plist_file"
        success "Launchd service installed. Use 'launchctl start com.fao.orchestrator' to start"
    else
        warning "Service installation not supported on this platform"
    fi
}

# Show usage
show_usage() {
    cat << EOF
FAO Dashboard Agent Orchestrator Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  start                 Start the orchestrator
  stop                  Stop the orchestrator
  restart               Restart the orchestrator
  status                Show orchestrator status
  monitor               Real-time monitoring (interactive)
  logs [--tail]         Show log files (optionally with recent entries)
  health                Perform health check
  report [format]       Generate report (html, csv, json)
  install-service       Install as system service
  help                  Show this help message

Examples:
  $0 start              # Start orchestrator
  $0 status             # Check if running and show status
  $0 monitor            # Real-time monitoring dashboard
  $0 logs --tail        # Show recent log entries
  $0 health             # Comprehensive health check
  $0 report html        # Generate HTML status report

For more detailed CLI options, use:
  python3 $CLI_SCRIPT --help

EOF
}

# Main execution
main() {
    local command="${1:-help}"
    
    # Setup logging
    setup_logging
    
    case "$command" in
        start)
            check_prerequisites
            validate_config
            start_orchestrator
            ;;
        stop)
            stop_orchestrator
            ;;
        restart)
            restart_orchestrator
            ;;
        status)
            show_status
            ;;
        monitor)
            monitor_orchestrator
            ;;
        logs)
            show_logs "${2:-}"
            ;;
        health)
            health_check
            ;;
        report)
            generate_report "${2:-html}"
            ;;
        install-service)
            install_service
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            error_exit "Unknown command: $command. Use '$0 help' for usage information."
            ;;
    esac
}

# Run main function with all arguments
main "$@"