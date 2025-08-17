# FAO Food Price Index Dashboard - Deployment Guide

This guide covers deployment options for the FAO Food Price Index Dashboard across multiple platforms and environments.

## 📋 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run with Streamlit config
streamlit run app.py

# Access at: http://localhost:8501
```

### Production Deployment
Choose from multiple deployment options below based on your infrastructure needs.

## 🔧 Configuration Files

### Core Configuration

| File | Purpose | Platform |
|------|---------|----------|
| `.streamlit/config.toml` | Streamlit app configuration | All |
| `Procfile` | Process definition | Heroku, Railway |
| `setup.sh` | Environment setup script | All |
| `runtime.txt` | Python version specification | Heroku, others |
| `requirements.txt` | Python dependencies | All |

### Platform-Specific

| File | Purpose | Platform |
|------|---------|----------|
| `Dockerfile` | Container configuration | Docker, Kubernetes |
| `docker-compose.yml` | Multi-container setup | Docker Compose |
| `app.yaml` | App Engine configuration | Google Cloud |
| `.streamlit/secrets.toml.example` | Secrets template | All |

## 🚀 Deployment Platforms

### 1. Streamlit Community Cloud
**Recommended for quick demos and prototypes**

1. **Setup Repository**
   ```bash
   git add .
   git commit -m "Deploy FAO Dashboard to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repository
   - Select `app.py` as main file
   - Deploy automatically

3. **Configuration**
   - Streamlit Cloud automatically uses `.streamlit/config.toml`
   - Add secrets via web interface if needed

### 2. Heroku
**Great for production deployments with custom domains**

1. **Prerequisites**
   ```bash
   # Install Heroku CLI
   # Create Heroku account
   heroku login
   ```

2. **Deploy**
   ```bash
   # Create Heroku app
   heroku create fao-dashboard-your-name
   
   # Set Python version
   echo "python-3.12.0" > runtime.txt
   
   # Deploy
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

3. **Configure**
   ```bash
   # Set environment variables
   heroku config:set STREAMLIT_SERVER_HEADLESS=true
   heroku config:set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   
   # Scale web dyno
   heroku ps:scale web=1
   ```

### 3. Railway
**Modern platform with great developer experience**

1. **Deploy**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and deploy
   railway login
   railway init
   railway up
   ```

2. **Configure**
   - Railway automatically detects `Procfile`
   - Uses `setup.sh` for environment preparation
   - Automatic HTTPS and custom domains

### 4. Google Cloud Platform
**Enterprise-grade with high scalability**

1. **App Engine Deployment**
   ```bash
   # Install Google Cloud SDK
   gcloud init
   gcloud auth login
   
   # Deploy
   gcloud app deploy app.yaml
   ```

2. **Cloud Run Deployment**
   ```bash
   # Build and deploy container
   gcloud builds submit --tag gcr.io/PROJECT_ID/fao-dashboard
   gcloud run deploy --image gcr.io/PROJECT_ID/fao-dashboard --platform managed
   ```

### 5. Docker Deployment
**For containerized environments and self-hosting**

1. **Local Docker**
   ```bash
   # Build image
   docker build -t fao-dashboard .
   
   # Run container
   docker run -p 8501:8501 fao-dashboard
   ```

2. **Docker Compose**
   ```bash
   # Development
   docker-compose up
   
   # Production with nginx
   docker-compose --profile production up -d
   ```

3. **Kubernetes**
   ```bash
   # Create deployment (example)
   kubectl create deployment fao-dashboard --image=fao-dashboard:latest
   kubectl expose deployment fao-dashboard --port=8501 --type=LoadBalancer
   ```

### 6. DigitalOcean App Platform
**Balanced performance and pricing**

1. **Deploy**
   - Connect GitHub repository via DigitalOcean console
   - Select `Procfile` deployment method
   - Configure auto-deploy from main branch

### 7. Azure Container Instances
**Microsoft cloud platform**

1. **Deploy**
   ```bash
   # Build and push to Azure Container Registry
   az acr build --registry myregistry --image fao-dashboard .
   
   # Deploy to Container Instances
   az container create --resource-group myResourceGroup --name fao-dashboard --image myregistry.azurecr.io/fao-dashboard:latest
   ```

## ⚙️ Configuration Details

### Streamlit Configuration (`.streamlit/config.toml`)

**Key Settings:**
- **Theme**: FAO blue color scheme (#1f77b4)
- **Layout**: Wide mode default for better charts
- **Upload**: 500MB limit for large Excel files
- **Cache**: 1-hour TTL matching app cache
- **Security**: XSRF protection enabled

**Customization:**
```toml
[theme]
primaryColor = "#1f77b4"  # FAO Blue
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"
base = "light"
font = "sans serif"

[server]
maxUploadSize = 500  # MB
port = 8501
enableXsrfProtection = true
```

### Environment Variables

**Required:**
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

**Optional Performance:**
```bash
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
PYTHONUNBUFFERED=1
TZ=UTC
```

**Platform-Specific:**
```bash
# Heroku
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false

# Cloud platforms
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
```

## 🔒 Security Considerations

### Production Security
1. **Environment Variables**: Use platform secret management
2. **HTTPS**: Enable SSL/TLS for all deployments
3. **Access Control**: Implement authentication if needed
4. **Data Privacy**: FAO data is public, no sensitive information

### Security Headers
Configure reverse proxy (nginx) with security headers:
```nginx
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
```

## 📊 Performance Optimization

### Resource Requirements

| Deployment Type | RAM | CPU | Storage |
|----------------|-----|-----|---------|
| Development | 512MB | 1 vCPU | 1GB |
| Small Production | 1GB | 1 vCPU | 2GB |
| Medium Production | 2GB | 2 vCPU | 5GB |
| Large Production | 4GB | 4 vCPU | 10GB |

### Optimization Tips

1. **Caching**: Data pipeline uses intelligent caching
2. **Memory**: Configure appropriate instance sizes
3. **CDN**: Use CDN for static assets if hosting separately
4. **Database**: Consider external database for high-traffic scenarios

### Monitoring

1. **Health Checks**: All configurations include health endpoints
2. **Logs**: Structured logging for debugging
3. **Metrics**: Monitor response times and error rates
4. **Alerts**: Set up alerts for deployment failures

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce cache size
   export STREAMLIT_RUNNER_MAX_MESSAGE_SIZE=200
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
   ```

2. **Port Binding Issues**
   ```bash
   # Ensure correct port binding
   export PORT=8501
   export STREAMLIT_SERVER_PORT=$PORT
   ```

3. **CORS Errors**
   ```bash
   # Disable CORS for cloud deployments
   export STREAMLIT_SERVER_ENABLE_CORS=false
   ```

4. **FAO Data Access**
   ```bash
   # Check network connectivity
   curl -I https://www.fao.org/media/docs/worldfoodsituationlibraries/default-document-library/food_price_indices_data_aug.xls
   ```

### Debug Mode

Enable debug mode for troubleshooting:
```bash
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=true
export STREAMLIT_LOGGER_LEVEL=debug
```

### Log Analysis

Check logs for common patterns:
```bash
# Application logs
tail -f logs/app.log

# Deployment logs
heroku logs --tail  # Heroku
railway logs        # Railway
kubectl logs deployment/fao-dashboard  # Kubernetes
```

## 🔄 Continuous Deployment

### GitHub Actions Integration

The repository includes GitHub Actions workflows:
- **Quality Check**: Runs on push/PR
- **Cache Update**: Monthly data refresh
- **Deploy**: Automated deployment (configure per platform)

### Auto-Deploy Setup

1. **Heroku**
   ```bash
   heroku config:set GITHUB_TOKEN=your_token
   # Enable auto-deploy in Heroku dashboard
   ```

2. **Railway**
   - Auto-deploy configured via Railway dashboard
   - Deploys on push to main branch

3. **Streamlit Cloud**
   - Auto-deploy enabled by default
   - Monitors repository for changes

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers for multiple instances
- Consider caching layer (Redis) for shared cache
- Database for persistent data if needed

### Vertical Scaling
- Monitor memory usage during peak loads
- Scale instance sizes based on usage patterns
- Consider auto-scaling policies

### Cost Optimization
- Use spot instances where available
- Implement auto-shutdown for development environments
- Monitor resource usage and optimize

## 🎯 Platform Recommendations

| Use Case | Recommended Platform | Why |
|----------|---------------------|-----|
| **Demo/Prototype** | Streamlit Cloud | Free, easy setup |
| **Small Production** | Railway | Great UX, fair pricing |
| **Medium Production** | Heroku | Mature platform, add-ons |
| **Enterprise** | Google Cloud/AWS | Scalability, enterprise features |
| **Self-Hosted** | Docker + VPS | Full control, cost-effective |
| **High Traffic** | Kubernetes | Auto-scaling, reliability |

## 💡 Next Steps

1. **Choose Platform**: Select based on requirements and budget
2. **Configure Secrets**: Set up any required environment variables
3. **Test Deployment**: Deploy to staging environment first
4. **Monitor**: Set up monitoring and alerting
5. **Optimize**: Tune performance based on usage patterns
6. **Scale**: Plan for growth and traffic increases

## 📞 Support

For deployment issues:
1. Check logs and error messages
2. Validate configuration files
3. Test locally with same settings
4. Review platform-specific documentation
5. Check GitHub Issues for similar problems

The FAO Dashboard is designed to be deployment-friendly across multiple platforms with minimal configuration changes required!