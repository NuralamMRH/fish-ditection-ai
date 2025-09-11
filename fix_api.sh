#!/bin/bash

# Configuration
SERVER="root@89.116.134.190"

echo "🔧 Fixing Fish API service..."

# Install system dependencies
ssh $SERVER "apt update && apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev"

# Fix virtual environment and dependencies
ssh $SERVER "cd /opt/fish-api && \
    rm -rf venv && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install gunicorn flask flask-cors opencv-python-headless numpy torch torchvision requests Pillow"

# Upload API files
scp fish_analysis_api.py $SERVER:/opt/fish-api/
scp -r detector_v12 $SERVER:/opt/fish-api/
scp -r classification_rectangle_v7-1 $SERVER:/opt/fish-api/

# Fix service file
ssh $SERVER "cat > /etc/systemd/system/fish-api.service << 'EOL'
[Unit]
Description=Fish Analysis API
After=network.target

[Service]
User=root
WorkingDirectory=/opt/fish-api
Environment=\"PATH=/opt/fish-api/venv/bin\"
ExecStart=/opt/fish-api/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5004 fish_analysis_api:app --log-level debug
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL"

# Fix permissions
ssh $SERVER "chown -R root:root /opt/fish-api && \
    chmod -R 755 /opt/fish-api"

# Create log directory
ssh $SERVER "mkdir -p /var/log/fish-api && \
    chown root:root /var/log/fish-api"

# Reload and restart services
ssh $SERVER "systemctl daemon-reload && \
    systemctl enable fish-api && \
    systemctl restart fish-api && \
    systemctl status fish-api"

echo "✅ API service fixed!"
echo "Testing API..."
curl -k https://fishai.itrucksea.com/health 