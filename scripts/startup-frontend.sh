#!/bin/bash

# Frontend startup script for solana.bondrpc.com
# This script builds and deploys the frontend to nginx

FRONTEND_DIR="/home/kidgordones/0solana/solana2/frontend"
NGINX_ROOT="/var/www/solana-bondrpc-frontend"

echo "[$(date)] Starting frontend deployment..."

# Build frontend
cd $FRONTEND_DIR
npm install --silent
npm run build

# Deploy to nginx
sudo rm -rf $NGINX_ROOT/*
sudo cp -r $FRONTEND_DIR/dist/* $NGINX_ROOT/

# Set proper permissions
sudo chown -R www-data:www-data $NGINX_ROOT

# Reload nginx
sudo nginx -t && sudo systemctl reload nginx

echo "[$(date)] Frontend deployment complete"
