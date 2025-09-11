#!/bin/bash

# Configuration
SERVER="root@89.116.134.190"
DOMAIN="fishai.itrucksea.com"

echo "🔧 Fixing complete setup..."

# 1. Fix directories and permissions
ssh $SERVER "mkdir -p /opt/fish-api/static /opt/fish-api/uploads && \
    chown -R www-data:www-data /opt/fish-api"

# 2. Install system dependencies
ssh $SERVER "apt update && \
    apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev"

# 3. Set up Python environment
ssh $SERVER "cd /opt/fish-api && \
    rm -rf venv && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install gunicorn flask flask-cors opencv-python-headless numpy torch torchvision requests Pillow"

# 4. Upload API files
scp fish_analysis_api.py $SERVER:/opt/fish-api/
scp -r detector_v12 $SERVER:/opt/fish-api/
scp -r classification_rectangle_v7-1 $SERVER:/opt/fish-api/

# 5. Create service file
ssh $SERVER "cat > /etc/systemd/system/fish-api.service << 'EOL'
[Unit]
Description=Fish Analysis API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/fish-api
Environment=PATH=/opt/fish-api/venv/bin
ExecStart=/opt/fish-api/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5004 fish_analysis_api:app
Restart=always

[Install]
WantedBy=multi-user.target
EOL"

# 6. Fix Nginx configuration
scp nginx.conf $SERVER:/etc/nginx/sites-available/fishai

# 7. Enable Nginx site and remove default
ssh $SERVER "rm -f /etc/nginx/sites-enabled/default && \
    ln -sf /etc/nginx/sites-available/fishai /etc/nginx/sites-enabled/ && \
    nginx -t && \
    systemctl restart nginx"

# 8. Start API service
ssh $SERVER "systemctl daemon-reload && \
    systemctl enable fish-api && \
    systemctl restart fish-api"

# 9. Set up SSL
ssh $SERVER "certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@itrucksea.com"

echo "✅ Setup complete! Testing services..."

# 10. Test services
ssh $SERVER "systemctl status nginx && systemctl status fish-api" 