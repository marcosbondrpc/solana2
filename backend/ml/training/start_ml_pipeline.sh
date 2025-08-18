#!/bin/bash

# MEV Arbitrage ML Pipeline Startup Script
# Production-grade deployment with health checks and monitoring

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ML_PIPELINE_DIR="/home/kidgordones/0solana/node/arbitrage-data-capture/ml-pipeline"
LOG_DIR="$ML_PIPELINE_DIR/logs"
MODEL_DIR="$ML_PIPELINE_DIR/models"
PYTHON_ENV="$ML_PIPELINE_DIR/venv"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed. Some features may not work."
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    else
        log_warning "No GPU detected. Models will run on CPU."
    fi
    
    # Check services
    if ! nc -z localhost 6379 2>/dev/null; then
        log_warning "Redis is not running on port 6379"
    fi
    
    if ! nc -z localhost 9092 2>/dev/null; then
        log_warning "Kafka is not running on port 9092"
    fi
    
    if ! nc -z localhost 9000 2>/dev/null; then
        log_warning "ClickHouse is not running on port 9000"
    fi
}

setup_environment() {
    log_info "Setting up Python environment..."
    
    cd "$ML_PIPELINE_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$PYTHON_ENV" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$PYTHON_ENV"
    fi
    
    # Activate virtual environment
    source "$PYTHON_ENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Create directories
    mkdir -p "$LOG_DIR" "$MODEL_DIR" "$ML_PIPELINE_DIR/cache" "$ML_PIPELINE_DIR/monitoring"
}

train_models() {
    log_info "Training ML models..."
    
    cd "$ML_PIPELINE_DIR"
    source "$PYTHON_ENV/bin/activate"
    
    # Run training
    python main.py --mode train --config config.yaml
    
    if [ $? -eq 0 ]; then
        log_info "Model training completed successfully"
    else
        log_error "Model training failed"
        exit 1
    fi
}

start_model_server() {
    log_info "Starting model server..."
    
    cd "$ML_PIPELINE_DIR"
    source "$PYTHON_ENV/bin/activate"
    
    # Check if server is already running
    if pgrep -f "model_server:app" > /dev/null; then
        log_warning "Model server is already running"
        return
    fi
    
    # Start server in background
    nohup python -m uvicorn src.model_server:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 4 \
        --loop uvloop \
        --log-level info \
        > "$LOG_DIR/model_server.log" 2>&1 &
    
    SERVER_PID=$!
    echo $SERVER_PID > "$ML_PIPELINE_DIR/model_server.pid"
    
    # Wait for server to start
    sleep 5
    
    # Check if server is running
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "Model server started successfully (PID: $SERVER_PID)"
    else
        log_error "Failed to start model server"
        exit 1
    fi
}

start_monitoring() {
    log_info "Starting monitoring service..."
    
    cd "$ML_PIPELINE_DIR"
    source "$PYTHON_ENV/bin/activate"
    
    # Check if monitor is already running
    if pgrep -f "model_monitor" > /dev/null; then
        log_warning "Monitoring service is already running"
        return
    fi
    
    # Start monitor in background
    nohup python -c "
import asyncio
import sys
sys.path.append('$ML_PIPELINE_DIR/src')
from model_monitor import ModelMonitor, MonitoringConfig
import yaml

with open('$ML_PIPELINE_DIR/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

monitor_config = MonitoringConfig(**config['monitoring'])
monitor = ModelMonitor(monitor_config)
asyncio.run(monitor.start_monitoring())
" > "$LOG_DIR/monitor.log" 2>&1 &
    
    MONITOR_PID=$!
    echo $MONITOR_PID > "$ML_PIPELINE_DIR/monitor.pid"
    
    log_info "Monitoring service started (PID: $MONITOR_PID)"
}

start_docker_services() {
    log_info "Starting Docker services..."
    
    cd "$ML_PIPELINE_DIR"
    
    # Start services with Docker Compose
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        log_info "Docker services started successfully"
        docker-compose ps
    else
        log_error "Failed to start Docker services"
        exit 1
    fi
}

check_system_health() {
    log_info "Checking system health..."
    
    # Check model server
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✓ Model server is healthy"
    else
        log_error "✗ Model server is not responding"
    fi
    
    # Check metrics
    if curl -f http://localhost:8000/metrics > /dev/null 2>&1; then
        METRICS=$(curl -s http://localhost:8000/metrics)
        log_info "✓ Metrics endpoint is working"
        echo "$METRICS" | python -m json.tool
    else
        log_warning "✗ Metrics endpoint is not responding"
    fi
    
    # Check Redis
    if redis-cli ping > /dev/null 2>&1; then
        log_info "✓ Redis is running"
    else
        log_warning "✗ Redis is not responding"
    fi
    
    # Check Kafka
    if nc -z localhost 9092 2>/dev/null; then
        log_info "✓ Kafka is running"
    else
        log_warning "✗ Kafka is not responding"
    fi
    
    # Check ClickHouse
    if nc -z localhost 9000 2>/dev/null; then
        log_info "✓ ClickHouse is running"
    else
        log_warning "✗ ClickHouse is not responding"
    fi
}

benchmark_performance() {
    log_info "Running performance benchmark..."
    
    cd "$ML_PIPELINE_DIR"
    source "$PYTHON_ENV/bin/activate"
    
    python main.py --benchmark
}

stop_services() {
    log_info "Stopping services..."
    
    # Stop model server
    if [ -f "$ML_PIPELINE_DIR/model_server.pid" ]; then
        PID=$(cat "$ML_PIPELINE_DIR/model_server.pid")
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            log_info "Stopped model server (PID: $PID)"
        fi
        rm "$ML_PIPELINE_DIR/model_server.pid"
    fi
    
    # Stop monitor
    if [ -f "$ML_PIPELINE_DIR/monitor.pid" ]; then
        PID=$(cat "$ML_PIPELINE_DIR/monitor.pid")
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            log_info "Stopped monitor (PID: $PID)"
        fi
        rm "$ML_PIPELINE_DIR/monitor.pid"
    fi
    
    # Stop Docker services
    if command -v docker-compose &> /dev/null; then
        cd "$ML_PIPELINE_DIR"
        docker-compose down
    fi
}

# Main execution
main() {
    case "${1:-}" in
        start)
            log_info "Starting ML Pipeline..."
            check_requirements
            setup_environment
            
            # Check if models exist
            if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR)" ]; then
                log_warning "No trained models found. Training models first..."
                train_models
            fi
            
            start_model_server
            start_monitoring
            check_system_health
            log_info "ML Pipeline started successfully!"
            ;;
            
        train)
            log_info "Training models..."
            check_requirements
            setup_environment
            train_models
            ;;
            
        docker)
            log_info "Starting with Docker..."
            start_docker_services
            ;;
            
        benchmark)
            log_info "Running benchmark..."
            setup_environment
            benchmark_performance
            ;;
            
        health)
            check_system_health
            ;;
            
        stop)
            stop_services
            log_info "Services stopped"
            ;;
            
        restart)
            stop_services
            sleep 2
            main start
            ;;
            
        logs)
            if [ -d "$LOG_DIR" ]; then
                tail -f "$LOG_DIR"/*.log
            else
                log_error "Log directory not found"
            fi
            ;;
            
        *)
            echo "Usage: $0 {start|train|docker|benchmark|health|stop|restart|logs}"
            echo ""
            echo "Commands:"
            echo "  start     - Start the ML pipeline (train if needed)"
            echo "  train     - Train models only"
            echo "  docker    - Start with Docker Compose"
            echo "  benchmark - Run performance benchmark"
            echo "  health    - Check system health"
            echo "  stop      - Stop all services"
            echo "  restart   - Restart all services"
            echo "  logs      - Tail log files"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"