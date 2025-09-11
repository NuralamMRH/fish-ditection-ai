#!/bin/bash

# Configuration
SERVER="root@89.116.134.190"
DOMAIN="fishai.itrucksea.com"

echo "🔧 Setting up Nginx configuration for $DOMAIN..."

# Create Nginx configuration directories and files
ssh $SERVER "mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled"

# Create main nginx.conf
ssh $SERVER "cat > /etc/nginx/nginx.conf << 'EOL'
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
EOL"

# Create site configuration
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
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    location /uploads/ {
        alias /var/www/fishai/uploads/;
        try_files \$uri \$uri/ =404;
    }
}
EOL"

# Create necessary directories
ssh $SERVER "mkdir -p /var/www/fishai/uploads && \
    chown -R www-data:www-data /var/www/fishai"

# Enable the site
ssh $SERVER "ln -sf /etc/nginx/sites-available/fishai /etc/nginx/sites-enabled/ && \
    rm -f /etc/nginx/sites-enabled/default"

# Test and restart Nginx
ssh $SERVER "nginx -t && \
    systemctl restart nginx"

# Install SSL certificate
ssh $SERVER "certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@itrucksea.com"

echo "✅ Nginx configuration completed!"
echo "Testing configuration..."
ssh $SERVER "nginx -t"

echo "
🎉 Nginx setup completed!
Your site should now be configured at: https://$DOMAIN
" 