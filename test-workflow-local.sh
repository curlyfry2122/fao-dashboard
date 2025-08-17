#!/bin/bash
# Local GitHub Actions workflow testing script using act tool
# This script helps test the update_cache.yml workflow locally before deployment

set -e

echo "🧪 FAO Dashboard - Local GitHub Actions Testing"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if act is installed
check_act_installation() {
    echo -e "${BLUE}🔍 Checking act installation...${NC}"
    
    if ! command -v act &> /dev/null; then
        echo -e "${RED}❌ act tool not found!${NC}"
        echo ""
        echo -e "${YELLOW}📥 To install act:${NC}"
        echo ""
        echo -e "${BLUE}macOS (Homebrew):${NC}"
        echo "  brew install act"
        echo ""
        echo -e "${BLUE}Linux (curl):${NC}"
        echo "  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
        echo ""
        echo -e "${BLUE}Windows (Chocolatey):${NC}"
        echo "  choco install act-cli"
        echo ""
        echo -e "${BLUE}Manual install:${NC}"
        echo "  Download from: https://github.com/nektos/act/releases"
        echo ""
        exit 1
    else
        ACT_VERSION=$(act --version)
        echo -e "${GREEN}✅ act is installed: ${ACT_VERSION}${NC}"
    fi
}

# Check Docker installation
check_docker() {
    echo -e "${BLUE}🐳 Checking Docker installation...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found! Act requires Docker to run.${NC}"
        echo "Install Docker from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker daemon not running! Please start Docker.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker is running${NC}"
}

# Setup test environment
setup_test_env() {
    echo -e "${BLUE}🔧 Setting up test environment...${NC}"
    
    # Create test cache directory
    mkdir -p .pipeline_cache_test
    
    # Create .env file for testing if it doesn't exist
    if [ ! -f .env ]; then
        echo -e "${YELLOW}📝 Creating .env file for testing...${NC}"
        cat > .env << EOF
# Test environment variables for act
GITHUB_TOKEN=fake_token_for_local_testing
PYTHONPATH=.
TZ=UTC
EOF
    fi
    
    # Create artifact directory
    mkdir -p /tmp/act-artifacts
    
    echo -e "${GREEN}✅ Test environment ready${NC}"
}

# List available workflows
list_workflows() {
    echo -e "${BLUE}📋 Available workflows:${NC}"
    echo ""
    
    if [ -d .github/workflows ]; then
        for workflow in .github/workflows/*.yml .github/workflows/*.yaml; do
            if [ -f "$workflow" ]; then
                basename "$workflow"
            fi
        done
    else
        echo -e "${RED}❌ No .github/workflows directory found${NC}"
        exit 1
    fi
    echo ""
}

# Test specific workflow
test_workflow() {
    local workflow_file="$1"
    local event_type="${2:-workflow_dispatch}"
    
    echo -e "${BLUE}🚀 Testing workflow: ${workflow_file}${NC}"
    echo -e "${BLUE}📅 Event type: ${event_type}${NC}"
    echo ""
    
    # Backup real cache if it exists
    if [ -d .pipeline_cache ]; then
        echo -e "${YELLOW}💾 Backing up existing cache...${NC}"
        cp -r .pipeline_cache .pipeline_cache_backup
    fi
    
    # Run the workflow with act
    echo -e "${GREEN}▶️ Starting workflow execution...${NC}"
    echo ""
    
    if [ "$event_type" = "schedule" ]; then
        # Test scheduled run
        act schedule -W "$workflow_file" --rm
    elif [ "$event_type" = "workflow_dispatch" ]; then
        # Test manual trigger with inputs
        cat > /tmp/workflow_dispatch_event.json << EOF
{
  "inputs": {
    "force_update": "true",
    "create_release": "false",
    "sheet_types": "Monthly"
  }
}
EOF
        act workflow_dispatch -W "$workflow_file" --rm --eventpath /tmp/workflow_dispatch_event.json
    else
        act "$event_type" -W "$workflow_file" --rm
    fi
    
    local exit_code=$?
    
    # Restore cache backup
    if [ -d .pipeline_cache_backup ]; then
        echo -e "${YELLOW}🔄 Restoring cache backup...${NC}"
        rm -rf .pipeline_cache
        mv .pipeline_cache_backup .pipeline_cache
    fi
    
    # Clean up test files
    rm -rf .pipeline_cache_test
    rm -f /tmp/workflow_dispatch_event.json
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Workflow test completed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}❌ Workflow test failed with exit code: $exit_code${NC}"
    fi
    
    return $exit_code
}

# Dry run - validate workflow syntax without execution
dry_run() {
    local workflow_file="$1"
    
    echo -e "${BLUE}🔍 Performing dry run validation...${NC}"
    echo ""
    
    # Use act's --dry-run flag to validate syntax
    act workflow_dispatch -W "$workflow_file" --dry-run
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Workflow syntax is valid!${NC}"
    else
        echo ""
        echo -e "${RED}❌ Workflow validation failed!${NC}"
    fi
    
    return $exit_code
}

# Clean up function
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    
    # Remove test files
    rm -rf .pipeline_cache_test
    rm -f /tmp/workflow_dispatch_event.json
    
    # Remove test artifacts
    rm -rf /tmp/act-artifacts
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Show help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  test [workflow] [event]  Test a specific workflow (default: update_cache.yml, workflow_dispatch)"
    echo "  list                     List available workflows"
    echo "  dry-run [workflow]       Validate workflow syntax without execution"
    echo "  setup                    Setup testing environment only"
    echo "  cleanup                  Clean up test files"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test                                    # Test update_cache.yml with workflow_dispatch"
    echo "  $0 test update_cache.yml schedule         # Test scheduled trigger"
    echo "  $0 test quality-check.yml push            # Test quality check on push"
    echo "  $0 dry-run update_cache.yml               # Validate syntax only"
    echo ""
    echo "Event types:"
    echo "  workflow_dispatch  Manual trigger (default)"
    echo "  schedule           Scheduled trigger (cron)"
    echo "  push              Push trigger"
    echo "  pull_request      PR trigger"
    echo ""
}

# Main script logic
main() {
    case "${1:-test}" in
        "setup")
            check_act_installation
            check_docker
            setup_test_env
            echo -e "${GREEN}🎉 Setup completed! You can now run workflow tests.${NC}"
            ;;
        "list")
            list_workflows
            ;;
        "test")
            check_act_installation
            check_docker
            setup_test_env
            
            local workflow="${2:-update_cache.yml}"
            local event="${3:-workflow_dispatch}"
            
            if [ ! -f ".github/workflows/$workflow" ]; then
                echo -e "${RED}❌ Workflow file not found: .github/workflows/$workflow${NC}"
                echo ""
                echo "Available workflows:"
                list_workflows
                exit 1
            fi
            
            test_workflow ".github/workflows/$workflow" "$event"
            ;;
        "dry-run")
            check_act_installation
            check_docker
            
            local workflow="${2:-update_cache.yml}"
            
            if [ ! -f ".github/workflows/$workflow" ]; then
                echo -e "${RED}❌ Workflow file not found: .github/workflows/$workflow${NC}"
                exit 1
            fi
            
            dry_run ".github/workflows/$workflow"
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Trap cleanup on script exit
trap cleanup EXIT

# Run main function with all arguments
main "$@"