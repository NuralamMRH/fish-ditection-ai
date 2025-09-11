#!/bin/bash

# Configuration
SERVER="root@89.116.134.190"
DOMAIN="fishai.itrucksea.com"

echo "🔧 Setting up Nginx and API service..."

# 1. Create Nginx configuration
echo "Creating Nginx site configuration..."
ssh $SERVER "cat > /etc/nginx/sites-available/fishai << 'EOL'
server {
    listen 80;
    server_name fishai.itrucksea.com;
    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:5004;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }

    location /static {
        alias /opt/fish-api/static;
    }

    location /uploads {
        alias /opt/fish-api/uploads;
    }
}
EOL"

# 2. Enable the site
echo "Enabling the site..."
ssh $SERVER "ln -sf /etc/nginx/sites-available/fishai /etc/nginx/sites-enabled/fishai"

# 3. Create directories
echo "Creating required directories..."
ssh $SERVER "mkdir -p /opt/fish-api/static /opt/fish-api/uploads"

# 4. Set up Python environment and dependencies
echo "Setting up Python environment..."
ssh $SERVER "cd /opt/fish-api && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install gunicorn flask opencv-python-headless numpy torch torchvision"

# 5. Upload API files
echo "Uploading API files..."
scp fish_analysis_api.py $SERVER:/opt/fish-api/
scp -r detector_v12 $SERVER:/opt/fish-api/
scp -r classification_rectangle_v7-1 $SERVER:/opt/fish-api/

# 6. Create service file
echo "Creating systemd service..."
ssh $SERVER "cat > /etc/systemd/system/fish-api.service << 'EOL'
[Unit]
Description=Fish Analysis API
After=network.target

[Service]
User=root
WorkingDirectory=/opt/fish-api
Environment=\"PATH=/opt/fish-api/venv/bin\"
ExecStart=/opt/fish-api/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5004 fish_analysis_api:app
Restart=always

[Install]
WantedBy=multi-user.target
EOL"

# 7. Set permissions
echo "Setting permissions..."
ssh $SERVER "chown -R www-data:www-data /opt/fish-api/static /opt/fish-api/uploads"

# 8. Reload and restart services
echo "Restarting services..."
ssh $SERVER "systemctl daemon-reload && \
    systemctl enable fish-api && \
    systemctl restart fish-api && \
    systemctl restart nginx"

# 9. Install and configure SSL
echo "Setting up SSL..."
ssh $SERVER "certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@itrucksea.com"

echo "✅ Setup complete! Please check https://$DOMAIN" 