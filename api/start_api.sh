#!/bin/bash
# Start the MEV API server

set -e

echo "🚀 Starting MEV API Server..."

# Set environment variables
export PYTHONPATH=/home/kidgordones/0solana/solana2/api:$PYTHONPATH
export PYTHONUNBUFFERED=1

# ClickHouse configuration
export CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-localhost}
export CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-8123}
export CLICKHOUSE_DB=${CLICKHOUSE_DB:-mev}
export CLICKHOUSE_USER=${CLICKHOUSE_USER:-default}
export CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD:-}

# Kafka configuration
export KAFKA_SERVERS=${KAFKA_SERVERS:-localhost:9092}

# Create necessary directories
mkdir -p /var/log/mev-api /tmp/mev_exports

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 4 \
        --loop uvloop \
        --access-log \
        --log-level info
else
    echo "Running locally"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install/upgrade dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Run the server
    echo "Starting server on http://0.0.0.0:8000"
    echo "API documentation available at http://0.0.0.0:8000/docs"
    
    exec python -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info
fi