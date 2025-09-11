#!/bin/bash

# Configuration
SERVER="root@89.116.134.190"
DOMAIN="fishai.itrucksea.com"

echo "🔒 Setting up SSL for $DOMAIN..."

# Install Certbot if not already installed
echo "📦 Installing Certbot..."
ssh $SERVER "apt update && apt install -y certbot python3-certbot-nginx"

# Ensure Nginx is running
echo "🌐 Ensuring Nginx is running..."
ssh $SERVER "systemctl start nginx"

# Create required directories
echo "📁 Creating required directories..."
ssh $SERVER "mkdir -p /var/www/fishai/uploads && chown -R www-data:www-data /var/www/fishai"

# Copy Nginx configuration
echo "📝 Updating Nginx configuration..."
scp nginx.conf $SERVER:/etc/nginx/sites-available/fishai

# Enable the site
echo "🔧 Enabling Nginx site..."
ssh $SERVER "ln -sf /etc/nginx/sites-available/fishai /etc/nginx/sites-enabled/fishai && \
    rm -f /etc/nginx/sites-enabled/default && \
    nginx -t && \
    systemctl reload nginx"

# Install SSL certificate
echo "🔒 Installing SSL certificate..."
ssh $SERVER "certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@itrucksea.com"

# Verify API service is running
echo "🚀 Verifying API service..."
ssh $SERVER "systemctl status fish-api || systemctl start fish-api"

echo "✅ Setup complete! Testing the domain..."
curl -k https://$DOMAIN/health

echo "
🎉 Deployment completed!
Your API should now be accessible at: https://$DOMAIN
" 