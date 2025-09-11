# Fish Analysis API Deployment Guide

## Ubuntu Server with Nginx Configuration

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [Server Setup](#server-setup)
3. [Project Deployment](#project-deployment)
4. [Environment Setup](#environment-setup)
5. [Nginx Configuration](#nginx-configuration)
6. [SSL Configuration](#ssl-configuration)
7. [Service Configuration](#service-configuration)
8. [Testing](#testing)
9. [Maintenance](#maintenance)
10. [Troubleshooting](#troubleshooting)

### Prerequisites

- Ubuntu Server (20.04 LTS or higher)
- SSH access to server (root@89.116.134.190)
- Domain name (optional but recommended)
- Basic knowledge of Linux commands

### Server Setup

1. **Update System**

```bash
apt update && apt upgrade -y
```

2. **Install Required Packages**

```bash
apt install -y python3-pip python3-venv nginx supervisor git
```

3. **Install System Dependencies**

```bash
apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### Project Deployment

1. **Create Project Directory**

```bash
mkdir -p /opt/fish-api
cd /opt/fish-api
```

2. **Clone/Copy Project Files**

```bash
# Option 1: Clone from repository (if available)
git clone <your-repo-url> .

# Option 2: Copy files using SCP
scp -r /path/to/local/project/* root@89.116.134.190:/opt/fish-api/
```

3. **Set Permissions**

```bash
chown -R www-data:www-data /opt/fish-api
chmod -R 755 /opt/fish-api
```

### Environment Setup

1. **Create Virtual Environment**

```bash
cd /opt/fish-api
python3 -m venv venv
source venv/bin/activate
```

2. **Install Python Dependencies**

```bash
pip install -r requirements.txt
pip install gunicorn  # For production server
```

3. **Test Dependencies**

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "from local_inference import LocalYOLOv12Fish; print('YOLO available')"
```

### Nginx Configuration

1. **Create Nginx Configuration**

```bash
nano /etc/nginx/sites-available/fish-api
```

2. **Add Configuration Content**

```nginx
server {
    listen 80;
    server_name 89.116.134.190;  # Replace with your domain if available

    client_max_body_size 16M;  # Match Flask config

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /opt/fish-api/static;
    }
}
```

3. **Enable Site Configuration**

```bash
ln -s /etc/nginx/sites-available/fish-api /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default  # Remove default site
nginx -t  # Test configuration
systemctl restart nginx
```

### Service Configuration

1. **Create Systemd Service**

```bash
nano /etc/systemd/system/fish-api.service
```

2. **Add Service Configuration**

```ini
[Unit]
Description=Fish Analysis API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/fish-api
Environment="PATH=/opt/fish-api/venv/bin"
ExecStart=/opt/fish-api/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:8000 fish_analysis_api:app
Restart=always

[Install]
WantedBy=multi-user.target
```

3. **Enable and Start Service**

```bash
systemctl daemon-reload
systemctl enable fish-api
systemctl start fish-api
```

### SSL Configuration (Optional but Recommended)

1. **Install Certbot**

```bash
apt install -y certbot python3-certbot-nginx
```

2. **Obtain SSL Certificate**

```bash
# If you have a domain:
certbot --nginx -d yourdomain.com

# For IP only (self-signed):
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/fish-api.key \
  -out /etc/ssl/certs/fish-api.crt
```

3. **Update Nginx Config for SSL**

```nginx
server {
    listen 443 ssl;
    server_name 89.116.134.190;

    ssl_certificate /etc/ssl/certs/fish-api.crt;
    ssl_certificate_key /etc/ssl/private/fish-api.key;

    # ... rest of the configuration ...
}
```

### Testing

1. **Check Service Status**

```bash
systemctl status fish-api
journalctl -u fish-api  # View logs
```

2. **Test API Endpoints**

```bash
# Health check
curl http://89.116.134.190/health

# Model info
curl http://89.116.134.190/models

# Test image analysis
curl -X POST -F "image=@test.jpg" http://89.116.134.190/analyze
```

### Maintenance

1. **Log Rotation**

```bash
nano /etc/logrotate.d/fish-api
```

Add:

```
/opt/fish-api/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload fish-api
    endscript
}
```

2. **Backup Script**

```bash
nano /opt/fish-api/backup.sh
```

Add:

```bash
#!/bin/bash
BACKUP_DIR="/opt/backups/fish-api"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/fish-api-$(date +%Y%m%d).tar.gz /opt/fish-api
find $BACKUP_DIR -type f -mtime +30 -delete
```

3. **Regular Updates**

```bash
# Update system
apt update && apt upgrade -y

# Update Python packages
source /opt/fish-api/venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Troubleshooting

1. **Check Logs**

```bash
# Nginx logs
tail -f /var/log/nginx/error.log
tail -f /var/log/nginx/access.log

# Application logs
journalctl -u fish-api -f

# System logs
dmesg | tail
```

2. **Common Issues**

- **502 Bad Gateway**

  ```bash
  systemctl restart fish-api
  systemctl status fish-api
  ```

- **Permission Issues**

  ```bash
  chown -R www-data:www-data /opt/fish-api
  chmod -R 755 /opt/fish-api
  ```

- **Memory Issues**
  ```bash
  free -m
  top
  ```

3. **Quick Fixes**

```bash
# Restart all services
systemctl restart nginx
systemctl restart fish-api

# Clear cache
rm -rf /opt/fish-api/static/cache/*

# Test configuration
nginx -t
```

### Security Recommendations

1. **Firewall Configuration**

```bash
ufw allow ssh
ufw allow 'Nginx Full'
ufw enable
```

2. **Rate Limiting in Nginx**
   Add to server block:

```nginx
limit_req_zone $binary_remote_addr zone=fish_api:10m rate=10r/s;
location / {
    limit_req zone=fish_api burst=20 nodelay;
    # ... rest of location config ...
}
```

3. **Secure Headers**
   Add to nginx config:

```nginx
add_header X-Frame-Options "SAMEORIGIN";
add_header X-XSS-Protection "1; mode=block";
add_header X-Content-Type-Options "nosniff";
add_header Content-Security-Policy "default-src 'self'";
```

For support or issues, please contact the development team.
