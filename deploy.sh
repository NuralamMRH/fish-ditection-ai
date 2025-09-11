#!/bin/bash

# Configuration
SERVER="root@89.116.134.190"
DEPLOY_PATH="/opt/fish-api"
LOCAL_PATH="."

echo "🚀 Starting deployment to $SERVER..."

# 1. Install system dependencies
echo "📦 Installing system dependencies..."
ssh $SERVER "apt update && apt install -y python3-pip python3-venv nginx supervisor git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev"

# 2. Create and prepare directories
echo "📁 Creating directories..."
ssh $SERVER "mkdir -p $DEPLOY_PATH/static $DEPLOY_PATH/logs"

# 3. Upload files
echo "📤 Uploading project files..."
# Main API file
scp fish_analysis_api.py $SERVER:$DEPLOY_PATH/
# YOLO detector
scp -r detector_v12 $SERVER:$DEPLOY_PATH/
# Classification model
scp -r classification_rectangle_v7-1 $SERVER:$DEPLOY_PATH/
# Requirements file
scp requirements.txt $SERVER:$DEPLOY_PATH/

# 4. Set up Python environment and install dependencies
echo "🐍 Setting up Python environment..."
ssh $SERVER "cd $DEPLOY_PATH && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn"

# 5. Set permissions
echo "🔒 Setting permissions..."
ssh $SERVER "chown -R www-data:www-data $DEPLOY_PATH && \
    chmod -R 755 $DEPLOY_PATH"

# 6. Create systemd service
echo "⚙️ Creating systemd service..."
cat << EOF | ssh $SERVER "cat > /etc/systemd/system/fish-api.service"
[Unit]
Description=Fish Analysis API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=$DEPLOY_PATH
Environment="PATH=$DEPLOY_PATH/venv/bin"
ExecStart=$DEPLOY_PATH/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:8000 fish_analysis_api:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 7. Configure Nginx
echo "🌐 Configuring Nginx..."
cat << EOF | ssh $SERVER "cat > /etc/nginx/sites-available/fish-api"
server {
    listen 80;
    server_name 89.116.134.190;

    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /static {
        alias $DEPLOY_PATH/static;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-XSS-Protection "1; mode=block";
    add_header X-Content-Type-Options "nosniff";
    add_header Content-Security-Policy "default-src 'self'";

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=fish_api:10m rate=10r/s;
    location / {
        limit_req zone=fish_api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
}
EOF

# 8. Enable and start services
echo "🚦 Starting services..."
ssh $SERVER "ln -sf /etc/nginx/sites-available/fish-api /etc/nginx/sites-enabled/ && \
    rm -f /etc/nginx/sites-enabled/default && \
    systemctl daemon-reload && \
    systemctl enable fish-api && \
    systemctl restart fish-api && \
    systemctl restart nginx"

# 9. Create backup script
echo "💾 Creating backup script..."
cat << EOF | ssh $SERVER "cat > $DEPLOY_PATH/backup.sh"
#!/bin/bash
BACKUP_DIR="/opt/backups/fish-api"
mkdir -p \$BACKUP_DIR
tar -czf \$BACKUP_DIR/fish-api-\$(date +%Y%m%d).tar.gz $DEPLOY_PATH
find \$BACKUP_DIR -type f -mtime +30 -delete
EOF

ssh $SERVER "chmod +x $DEPLOY_PATH/backup.sh"

# 10. Set up log rotation
echo "📝 Setting up log rotation..."
cat << EOF | ssh $SERVER "cat > /etc/logrotate.d/fish-api"
$DEPLOY_PATH/logs/*.log {
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
EOF

echo "✅ Deployment completed!"
echo "
API should now be accessible at:
http://89.116.134.190/

Test the API with:
curl http://89.116.134.190/health
curl http://89.116.134.190/models

To monitor the service:
ssh $SERVER 'systemctl status fish-api'
ssh $SERVER 'journalctl -u fish-api -f'
" 