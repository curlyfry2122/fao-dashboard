# GitHub Actions Workflows

This document describes the automated workflows for the FAO Food Price Index Dashboard.

## 📋 Available Workflows

### 1. Update Cache Workflow (`update_cache.yml`)

Automatically updates the FAO data cache on a monthly schedule with manual trigger capability.

#### 🚀 Triggers

- **Monthly Schedule**: 15th of each month at 12:00 UTC
- **Manual Trigger**: `workflow_dispatch` with optional parameters

#### 🎛️ Manual Trigger Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `force_update` | Force update even if cache is recent | `false` | `true`/`false` |
| `create_release` | Create release tag after successful update | `true` | `true`/`false` |
| `sheet_types` | Comma-separated list of data types | `Monthly,Annual` | `Monthly`, `Annual`, or both |

#### 🔄 Workflow Steps

1. **Environment Setup**
   - Checkout repository with full history
   - Setup Python 3.12 with pip caching
   - Install dependencies from requirements.txt

2. **Cache Status Check**
   - Evaluate current cache file ages
   - Determine if update is needed (>7 days old or force_update=true)
   - Create backup of existing cache

3. **Data Update**
   - Execute data pipeline for specified sheet types
   - Fetch fresh FAO data from official sources
   - Update pickle cache files in `.pipeline_cache/`
   - Validate data integrity

4. **Validation & Testing**
   - Run pickle file validation
   - Execute data pipeline tests
   - Ensure data quality and format consistency

5. **Commit & Release**
   - Commit updated cache files with detailed message
   - Create annotated release tag (e.g., `cache-update-2025-01-15`)
   - Upload cache files as GitHub artifacts

6. **Error Handling**
   - Restore cache backup on failure
   - Send failure notifications
   - Generate comprehensive workflow summary

#### 📊 Outputs

| Output | Description |
|--------|-------------|
| `cache-updated` | Boolean indicating successful cache update |
| `release-tag` | Generated release tag name |
| `data-summary` | Summary of processed data records |

### 2. Quality Check Workflow (`quality-check.yml`)

Runs automated testing and validation on code changes.

#### 🚀 Triggers

- **Push**: to `main` or `develop` branches
- **Pull Request**: targeting `main` branch

#### 🧪 Test Matrix

- Python versions: 3.9, 3.10, 3.11
- Operating System: Ubuntu Latest

## 🛠️ Local Testing with Act

### Prerequisites

1. **Install Act Tool**
   ```bash
   # macOS
   brew install act
   
   # Linux
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   
   # Windows
   choco install act-cli
   ```

2. **Install Docker**
   - Download from: https://docs.docker.com/get-docker/
   - Ensure Docker daemon is running

### 🧪 Testing Commands

```bash
# Setup testing environment
./test-workflow-local.sh setup

# Test cache update workflow (manual trigger)
./test-workflow-local.sh test update_cache.yml workflow_dispatch

# Test scheduled cache update
./test-workflow-local.sh test update_cache.yml schedule

# Validate workflow syntax only
./test-workflow-local.sh dry-run update_cache.yml

# List available workflows
./test-workflow-local.sh list

# Clean up test files
./test-workflow-local.sh cleanup
```

### 🔧 Test Configuration

- **Configuration File**: `.actrc`
- **Test Script**: `test-workflow-local.sh`
- **Environment**: `.env` (auto-created for testing)

The test script automatically:
- Backs up existing cache files
- Creates isolated test environment
- Restores original state after testing
- Provides colored output for easy debugging

## 📈 Monitoring & Maintenance

### 🔍 Workflow Status

Monitor workflow status via:
- GitHub Actions tab in repository
- Release tags for successful cache updates
- Workflow artifacts containing cache files

### 📧 Notifications

Failed workflows trigger:
- GitHub notification to repository watchers
- Workflow summary with failure details
- Option to extend with email/Slack notifications

### 🛠️ Troubleshooting

#### Common Issues

1. **FAO Data Source Unavailable**
   - Workflow continues with existing cache
   - Manual retry available via workflow_dispatch
   - Fallback URLs configured in data_fetcher.py

2. **Cache Validation Failures**
   - Automatic rollback to previous cache
   - Detailed error logs in workflow summary
   - Manual investigation via downloaded artifacts

3. **Git Push Failures**
   - Check repository permissions
   - Verify GITHUB_TOKEN has write access
   - Review branch protection rules

#### 🔧 Manual Recovery

```bash
# Force update cache locally
python -c "
from data_pipeline import DataPipeline
pipeline = DataPipeline(sheet_name='Monthly', cache_ttl_hours=0)
df = pipeline.run()
print(f'Updated: {len(df)} records')
"

# Commit manual cache update
git add .pipeline_cache/*.pkl
git commit -m "Manual cache update - $(date)"
git push
```

### 📅 Schedule Management

#### Modify Update Frequency

Edit `.github/workflows/update_cache.yml`:

```yaml
schedule:
  # Monthly on 15th at 12:00 UTC
  - cron: '0 12 15 * *'
  
  # Weekly on Mondays at 09:00 UTC
  # - cron: '0 9 * * 1'
  
  # Daily at 06:00 UTC
  # - cron: '0 6 * * *'
```

#### Time Zone Considerations

- All cron schedules use UTC
- Consider FAO data publication schedule
- Account for server maintenance windows

## 🚀 Deployment Integration

### Streamlit Cloud

Cache updates automatically trigger deployment refresh on platforms that monitor the repository.

### Custom Deployment

Use workflow outputs in deployment pipelines:

```yaml
- name: Deploy if cache updated
  if: needs.update-cache.outputs.cache-updated == 'true'
  run: |
    echo "Deploying with fresh cache..."
    # Add deployment commands
```

### Docker Integration

Updated cache files are automatically included in Docker builds that use the repository as context.

## 🔒 Security Considerations

### Permissions

- Workflows use `GITHUB_TOKEN` with repository scope
- No external secrets required for basic operation
- Optional: Configure additional secrets for notifications

### Data Privacy

- FAO data is public domain
- Cache files contain no sensitive information
- Workflow logs may contain data summaries (non-sensitive)

### Best Practices

- Regular review of workflow permissions
- Monitor for unusual cache update patterns
- Keep act tool and Docker updated for local testing