#!/bin/bash

# Setup script for FAO Food Price Index Dashboard deployment
# This script prepares the environment for Streamlit deployment on various platforms

echo "🚀 Setting up FAO Food Price Index Dashboard..."

# Create necessary directories
mkdir -p ~/.streamlit/
mkdir -p .pipeline_cache
mkdir -p cache
mkdir -p logs

# Copy Streamlit configuration
cp .streamlit/config.toml ~/.streamlit/config.toml

# Create Streamlit credentials file for deployment (if needed)
echo "[general]
email = \"noreply@fao-dashboard.com\"
" > ~/.streamlit/credentials.toml

# Set environment variables for deployment
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

# Deployment platform detection and optimization
if [ -n "$DYNO" ]; then
    echo "📦 Detected Heroku deployment"
    # Heroku-specific optimizations
    export STREAMLIT_SERVER_ENABLE_CORS=false
    export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
    export STREAMLIT_SERVER_HEADLESS=true
elif [ -n "$RAILWAY_ENVIRONMENT" ]; then
    echo "🚂 Detected Railway deployment"
    # Railway-specific optimizations
    export STREAMLIT_SERVER_HEADLESS=true
elif [ -n "$VERCEL" ]; then
    echo "▲ Detected Vercel deployment"
    # Vercel-specific optimizations
    export STREAMLIT_SERVER_HEADLESS=true
elif [ -n "$RENDER" ]; then
    echo "🎨 Detected Render deployment"
    # Render-specific optimizations
    export STREAMLIT_SERVER_HEADLESS=true
else
    echo "🖥️ Local or generic deployment detected"
fi

# Performance optimizations
echo "⚡ Applying performance optimizations..."

# Set Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Memory management for cloud deployments
if [ -n "$DYNO" ] || [ -n "$RAILWAY_ENVIRONMENT" ]; then
    # Cloud deployment - optimize for limited memory
    export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200  # Reduced for cloud limits
    export STREAMLIT_RUNNER_MAX_MESSAGE_SIZE=200
else
    # Local/high-memory deployment
    export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
    export STREAMLIT_RUNNER_MAX_MESSAGE_SIZE=500
fi

# Create cache directories with proper permissions
chmod 755 .pipeline_cache 2>/dev/null || true
chmod 755 cache 2>/dev/null || true
chmod 755 logs 2>/dev/null || true

# Initialize empty cache files if they don't exist
touch .pipeline_cache/.gitkeep
touch cache/.gitkeep
touch logs/.gitkeep

# Health check setup
echo "🔍 Setting up health monitoring..."

# Create a simple health check endpoint data
echo "{
  \"status\": \"healthy\",
  \"service\": \"FAO Food Price Index Dashboard\",
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"version\": \"1.0.0\",
  \"environment\": \"${ENVIRONMENT:-production}\"
}" > health.json

# Pre-warm cache if possible (non-blocking)
echo "🔥 Pre-warming application cache..."
(
    timeout 30 python3 -c "
try:
    from data_pipeline import DataPipeline
    print('📊 Pre-loading data pipeline...')
    pipeline = DataPipeline(sheet_name='Monthly', cache_ttl_hours=24)
    # Attempt to pre-load data, but don't fail if FAO is unavailable
    pipeline.run()
    print('✅ Cache pre-warming completed')
except Exception as e:
    print(f'⚠️ Cache pre-warming failed (normal for first deployment): {str(e)}')
" 2>/dev/null
) &

# Install any missing system dependencies (if needed)
if command -v apt-get >/dev/null 2>&1; then
    echo "🔧 Installing system dependencies..."
    # For Debian/Ubuntu systems (some cloud platforms)
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y --no-install-recommends \
        fonts-liberation \
        libfontconfig1 \
        libglib2.0-0 \
        libgssapi-krb5-2 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libxss1 \
        libxtst6 >/dev/null 2>&1 || true
fi

# Set timezone to UTC for consistent data handling
export TZ=UTC

echo "✅ FAO Dashboard setup completed successfully!"
echo "🌐 Dashboard will be available on port ${PORT:-8501}"
echo "📊 Ready to serve FAO Food Price Index data"

# Optional: Display system information for debugging
if [ "${STREAMLIT_GLOBAL_DEVELOPMENT_MODE}" = "true" ]; then
    echo "🐛 Development mode - System information:"
    echo "   Python version: $(python3 --version 2>/dev/null || echo 'Not available')"
    echo "   Streamlit version: $(python3 -c 'import streamlit; print(streamlit.__version__)' 2>/dev/null || echo 'Not available')"
    echo "   Working directory: $(pwd)"
    echo "   Available memory: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'Unknown')"
    echo "   Available disk: $(df -h . 2>/dev/null | tail -1 | awk '{print $4}' || echo 'Unknown')"
fi