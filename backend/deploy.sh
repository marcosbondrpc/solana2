#!/bin/bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="${SCRIPT_DIR}/.env"
DOCKER_COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_warn "Environment file not found. Creating from example..."
        cp "${ENV_FILE}.example" "$ENV_FILE"
        log_warn "Please edit ${ENV_FILE} with your configuration"
        exit 1
    fi
    
    log_info "All requirements met"
}

build_services() {
    log_info "Building services..."
    
    # Build each service
    for service in api-gateway rpc-probe validator-agent jito-probe geyser-probe metrics controls; do
        log_info "Building ${service}..."
        cd "${SCRIPT_DIR}/${service}"
        
        # Install dependencies
        npm ci --production=false
        
        # Build TypeScript
        npm run build
        
        # Create Dockerfile if not exists
        if [ ! -f "Dockerfile" ]; then
            cat > Dockerfile << 'EOF'
FROM node:20-alpine

WORKDIR /app

# Install production dependencies
COPY package*.json ./
RUN npm ci --production

# Copy built application
COPY dist ./dist

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

USER nodejs

CMD ["node", "dist/index.js"]
EOF
        fi
    done
    
    cd "$SCRIPT_DIR"
    log_info "Services built successfully"
}

create_certificates() {
    log_info "Creating self-signed certificates for development..."
    
    CERT_DIR="${SCRIPT_DIR}/certs"
    mkdir -p "$CERT_DIR"
    
    if [ ! -f "${CERT_DIR}/ca.crt" ]; then
        # Generate CA key and certificate
        openssl genrsa -out "${CERT_DIR}/ca.key" 4096
        openssl req -new -x509 -days 365 -key "${CERT_DIR}/ca.key" -out "${CERT_DIR}/ca.crt" \
            -subj "/C=US/ST=CA/L=SF/O=Solana/CN=Solana CA"
        
        # Generate server key and certificate
        openssl genrsa -out "${CERT_DIR}/server.key" 4096
        openssl req -new -key "${CERT_DIR}/server.key" -out "${CERT_DIR}/server.csr" \
            -subj "/C=US/ST=CA/L=SF/O=Solana/CN=localhost"
        openssl x509 -req -days 365 -in "${CERT_DIR}/server.csr" -CA "${CERT_DIR}/ca.crt" \
            -CAkey "${CERT_DIR}/ca.key" -CAcreateserial -out "${CERT_DIR}/server.crt"
        
        # Generate client certificates for mTLS
        for client in admin-client operator-client; do
            openssl genrsa -out "${CERT_DIR}/${client}.key" 4096
            openssl req -new -key "${CERT_DIR}/${client}.key" -out "${CERT_DIR}/${client}.csr" \
                -subj "/C=US/ST=CA/L=SF/O=Solana/CN=${client}"
            openssl x509 -req -days 365 -in "${CERT_DIR}/${client}.csr" -CA "${CERT_DIR}/ca.crt" \
                -CAkey "${CERT_DIR}/ca.key" -CAcreateserial -out "${CERT_DIR}/${client}.crt"
        done
        
        log_info "Certificates created in ${CERT_DIR}"
    else
        log_info "Certificates already exist"
    fi
}

setup_nginx() {
    log_info "Setting up Nginx configuration..."
    
    NGINX_DIR="${SCRIPT_DIR}/nginx"
    mkdir -p "$NGINX_DIR"
    
    cat > "${NGINX_DIR}/nginx.conf" << 'EOF'
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    gzip on;
    gzip_disable "msie6";
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss 
               application/rss+xml application/atom+xml image/svg+xml 
               text/x-js text/x-cross-domain-policy application/x-font-ttf 
               application/x-font-opentype application/vnd.ms-fontobject 
               image/x-icon;

    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;

    # Upstream servers
    upstream api_gateway {
        least_conn;
        server api-gateway:3000 max_fails=3 fail_timeout=30s;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name _;

        ssl_certificate /etc/nginx/ssl/server.crt;
        ssl_certificate_key /etc/nginx/ssl/server.key;
        ssl_client_certificate /etc/nginx/ssl/ca.crt;
        ssl_verify_client optional;

        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "same-origin" always;

        # API Gateway
        location /api {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://api_gateway;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Client-Cert $ssl_client_cert;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 86400;
        }

        # Auth endpoints with stricter rate limiting
        location /auth {
            limit_req zone=auth burst=2 nodelay;
            
            proxy_pass http://api_gateway;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket support
        location /ws {
            proxy_pass http://api_gateway;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 86400;
        }

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://api_gateway/health;
        }
    }
}
EOF
    
    # Link SSL certificates
    ln -sf "${SCRIPT_DIR}/certs" "${NGINX_DIR}/ssl"
    
    log_info "Nginx configuration created"
}

start_services() {
    log_info "Starting services..."
    
    cd "$SCRIPT_DIR"
    
    # Start infrastructure services first
    docker-compose up -d redis
    sleep 5
    
    # Start backend services
    docker-compose up -d
    
    log_info "Services started"
}

check_health() {
    log_info "Checking service health..."
    
    sleep 10
    
    # Check each service
    services=("api-gateway:3000" "rpc-probe:3010" "validator-agent:3020" 
              "jito-probe:3030" "geyser-probe:3040" "metrics:3050" "controls:3060")
    
    for service in "${services[@]}"; do
        name="${service%%:*}"
        port="${service##*:}"
        
        if curl -f -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            log_info "${name} is healthy"
        else
            log_warn "${name} health check failed"
        fi
    done
}

main() {
    log_info "Starting Solana Backend Deployment"
    
    check_requirements
    build_services
    create_certificates
    setup_nginx
    start_services
    check_health
    
    log_info "Deployment complete!"
    log_info "API Gateway: https://localhost"
    log_info "Metrics: http://localhost:3050/metrics"
    log_warn "Remember to configure your .env file with production values"
}

# Run main function
main "$@"