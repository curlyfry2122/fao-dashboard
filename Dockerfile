# Dockerfile for FAO Food Price Index Dashboard
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
    TZ=UTC

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash streamlit

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    fonts-liberation \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/streamlit/.local

# Make sure scripts in .local are usable
ENV PATH=/home/streamlit/.local/bin:$PATH

# Copy application code
COPY --chown=streamlit:streamlit . .

# Create necessary directories
RUN mkdir -p .pipeline_cache cache logs && \
    chown -R streamlit:streamlit .pipeline_cache cache logs

# Copy Streamlit configuration
RUN mkdir -p /home/streamlit/.streamlit && \
    cp .streamlit/config.toml /home/streamlit/.streamlit/ && \
    chown -R streamlit:streamlit /home/streamlit/.streamlit

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8501/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Switch to non-root user
USER streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /app/healthcheck.sh

# Set default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Build instructions:
# docker build -t fao-dashboard .
# docker run -p 8501:8501 fao-dashboard

# For development with volume mounts:
# docker run -p 8501:8501 -v $(pwd):/app fao-dashboard

# For production with docker-compose:
# docker-compose up -d