# Deployment Guide for BetaControlAPI

## Replit Deployment

### 1. System Requirements for Replit

For optimal performance on Replit, choose a plan that provides:
- At least 4GB RAM (Replit Pro or higher recommended)
- 4 vCPUs
- Always-on capability
- Higher network bandwidth allocation

### 2. Quick Setup Steps

1. **Create New Repl**
   - Choose "Python" as your template
   - Import from GitHub: `https://github.com/WickedDevTeam/BetaControlAPI`

2. **Configure Environment**
   Add these secrets in Replit's "Secrets" tab:
   ```
   FLASK_ENV=production
   DEBUG=False
   PRODUCTION=True
   PORT=443  # Replit uses 443 for HTTPS
   HOST=0.0.0.0
   MAX_CONCURRENT_USERS=50
   RATE_LIMIT=1000
   ENABLE_CACHING=True
   ```

3. **Update `.replit` File**
   ```
   run = "python3 backend/app.py"
   language = "python3"
   entrypoint = "backend/app.py"
   hidden = [".config", "**/__pycache__", "**/.pytest_cache"]
   
   [nix]
   channel = "stable-23_05"
   
   [deployment]
   run = ["sh", "-c", "python3 backend/app.py"]
   deploymentTarget = "cloudrun"
   
   [env]
   PYTHONPATH = "${PYTHONPATH}:${REPL_HOME}"
   ```

4. **Create `replit.nix` File**
   ```nix
   { pkgs }: {
     deps = [
       pkgs.python39
       pkgs.python39Packages.pip
       pkgs.python39Packages.flask
       pkgs.python39Packages.opencv4
       pkgs.python39Packages.numpy
       pkgs.python39Packages.pillow
       pkgs.cmake
       pkgs.gcc
     ];
   }
   ```

### 3. Performance Optimizations

1. **Memory Management**
   ```python
   # Add to config.py
   UPLOAD_MAX_MEMORY = 512 * 1024 * 1024  # 512MB max memory usage
   GC_INTERVAL = 100  # Garbage collect every 100 requests
   ```

2. **Caching Strategy**
   ```python
   # Add to config.py
   CACHE_TYPE = "filesystem"  # Use filesystem caching
   CACHE_DIR = "/tmp/flask_cache"
   CACHE_DEFAULT_TIMEOUT = 600  # 10 minutes
   ```

3. **Request Handling**
   ```python
   # Add to config.py
   MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB max file size
   UPLOAD_EXTENSIONS = ['.jpg', '.jpeg', '.png']  # Restrict file types
   ```

### 4. Monitoring Setup

1. **Health Check Endpoint**
   ```python
   @app.route('/health')
   def health_check():
       return jsonify({
           'status': 'healthy',
           'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
           'cpu_percent': psutil.cpu_percent()
       })
   ```

2. **Error Alerts**
   ```python
   # Add to config.py
   DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')  # Add to Replit secrets
   ```

### 5. Scaling Guidelines

For your Replit deployment:
- Start with 50 concurrent users limit
- Monitor memory usage (should stay under 80% of available RAM)
- Set up auto-restart if memory exceeds 90%
- Implement request queuing for heavy processing tasks

### 6. Cost Management

Recommended Replit Plan:
- Replit Pro ($10/month) for development/testing
- Replit Teams Pro ($20/month) for production
  - Includes better performance
  - Higher resource limits
  - Always-on capability
  - Custom domains

### 7. Maintenance Tasks

Daily:
- Monitor error logs
- Check resource usage
- Clear temporary files

Weekly:
- Review performance metrics
- Update model files if needed
- Backup configuration

### 8. Security Notes

1. **API Protection**
   ```python
   # Add to config.py
   API_KEY_REQUIRED = True
   API_KEYS = os.getenv('ALLOWED_API_KEYS', '').split(',')
   ```

2. **Rate Limiting**
   ```python
   # Add to config.py
   RATE_LIMIT_ENABLED = True
   RATE_LIMIT_STORAGE_URL = "memory://"  # Use memory for rate limiting
   ```

### 9. Troubleshooting

Common issues and solutions:
1. Memory errors: Increase garbage collection frequency
2. Slow processing: Reduce max concurrent users
3. Timeouts: Implement request queuing
4. High CPU usage: Enable caching

### 10. Backup Strategy

1. **Configuration Backup**
   - Store all secrets in a secure location
   - Keep a local copy of configuration files

2. **Data Backup**
   - Use Replit's built-in Git integration
   - Regular commits of configuration changes
   - Store model files in separate storage 