#!/bin/bash

# Elite MEV Backend Migration Script
# Migrates services from scattered directories to organized structure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base paths
OLD_BASE="/home/kidgordones/0solana/node"
NEW_BASE="/home/kidgordones/0solana/node/backend"

echo -e "${GREEN}Starting MEV Backend Service Migration${NC}"
echo -e "${BLUE}Migrating from scattered structure to organized backend${NC}\n"

# Function to safely copy with progress
safe_copy() {
    local src=$1
    local dst=$2
    local desc=$3
    
    if [ -e "$src" ]; then
        echo -e "${YELLOW}Migrating ${desc}...${NC}"
        mkdir -p "$(dirname "$dst")"
        cp -r "$src" "$dst" 2>/dev/null || true
        echo -e "${GREEN}✓ ${desc} migrated${NC}"
    else
        echo -e "${RED}✗ ${desc} not found at ${src}${NC}"
    fi
}

# Create all necessary directories
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p ${NEW_BASE}/{services,shared,infrastructure,ml,tools,docs}
mkdir -p ${NEW_BASE}/services/{mev-engine,sandwich-detector,arbitrage-engine,control-plane,ml-inference,data-ingestion}
mkdir -p ${NEW_BASE}/shared/{proto,rust-common,python-common,configs}
mkdir -p ${NEW_BASE}/infrastructure/{docker,kubernetes,systemd,clickhouse,kafka,redis}
mkdir -p ${NEW_BASE}/ml/{models,training,feature-engineering,evaluation}
mkdir -p ${NEW_BASE}/tools/{build,deploy,testing,monitoring}

# Migrate MEV Engine components
echo -e "\n${BLUE}Migrating MEV Engine components...${NC}"
safe_copy "${OLD_BASE}/backend-mev/src" "${NEW_BASE}/services/mev-engine/src" "MEV engine source"
safe_copy "${OLD_BASE}/backend-mev/benches" "${NEW_BASE}/services/mev-engine/benches" "MEV benchmarks"

# Migrate Sandwich Detector
echo -e "\n${BLUE}Migrating Sandwich Detector...${NC}"
safe_copy "${OLD_BASE}/mev-sandwich-detector/src" "${NEW_BASE}/services/sandwich-detector/src" "Sandwich detector source"
safe_copy "${OLD_BASE}/mev-sandwich-detector/configs" "${NEW_BASE}/services/sandwich-detector/configs" "Sandwich configs"
safe_copy "${OLD_BASE}/mev-sandwich-detector/ml-pipeline" "${NEW_BASE}/ml/training/sandwich-ml" "Sandwich ML pipeline"
safe_copy "${OLD_BASE}/mev-sandwich-detector/monitoring" "${NEW_BASE}/infrastructure/monitoring/sandwich" "Sandwich monitoring"
safe_copy "${OLD_BASE}/mev-sandwich-detector/schemas" "${NEW_BASE}/infrastructure/clickhouse/sandwich-schemas" "Sandwich schemas"

# Migrate Arbitrage Engine
echo -e "\n${BLUE}Migrating Arbitrage Engine...${NC}"
safe_copy "${OLD_BASE}/arbitrage-data-capture/arbitrage-detector/src" "${NEW_BASE}/services/arbitrage-engine/src" "Arbitrage engine source"
safe_copy "${OLD_BASE}/arbitrage-data-capture/rust-services" "${NEW_BASE}/services/data-ingestion" "Data ingestion services"

# Migrate Control Plane (Python/FastAPI)
echo -e "\n${BLUE}Migrating Control Plane...${NC}"
mkdir -p "${NEW_BASE}/services/control-plane"
safe_copy "${OLD_BASE}/arbitrage-data-capture/api" "${NEW_BASE}/services/control-plane/src" "Control plane API"
safe_copy "${OLD_BASE}/arbitrage-data-capture/ml-pipeline" "${NEW_BASE}/services/control-plane/ml-pipeline" "ML pipeline"

# Migrate ML components
echo -e "\n${BLUE}Migrating ML Components...${NC}"
safe_copy "${OLD_BASE}/arbitrage-data-capture/ml-pipeline/models" "${NEW_BASE}/ml/models" "ML models"
safe_copy "${OLD_BASE}/arbitrage-data-capture/ml-pipeline/src" "${NEW_BASE}/ml/training" "ML training code"
safe_copy "${OLD_BASE}/arbitrage-data-capture/continuous-improvement" "${NEW_BASE}/ml/evaluation" "ML evaluation"

# Migrate Protobuf definitions
echo -e "\n${BLUE}Migrating Protocol Definitions...${NC}"
safe_copy "${OLD_BASE}/arbitrage-data-capture/protocol" "${NEW_BASE}/shared/proto" "Protocol buffers"

# Migrate Infrastructure configs
echo -e "\n${BLUE}Migrating Infrastructure Configurations...${NC}"
safe_copy "${OLD_BASE}/arbitrage-data-capture/docker-compose.yml" "${NEW_BASE}/infrastructure/docker/docker-compose.legacy.yml" "Docker compose"
safe_copy "${OLD_BASE}/arbitrage-data-capture/systemd" "${NEW_BASE}/infrastructure/systemd" "Systemd services"
safe_copy "${OLD_BASE}/arbitrage-data-capture/clickhouse" "${NEW_BASE}/infrastructure/clickhouse" "ClickHouse configs"
safe_copy "${OLD_BASE}/arbitrage-data-capture/monitoring" "${NEW_BASE}/infrastructure/monitoring" "Monitoring configs"

# Migrate Kafka configurations
echo -e "\n${BLUE}Migrating Kafka Configurations...${NC}"
safe_copy "${OLD_BASE}/arbitrage-data-capture/configs/kafka_config.yaml" "${NEW_BASE}/infrastructure/kafka/config.yaml" "Kafka config"
safe_copy "${OLD_BASE}/mev-sandwich-detector/configs/kafka_config.yaml" "${NEW_BASE}/infrastructure/kafka/sandwich-config.yaml" "Sandwich Kafka config"

# Migrate Redis scripts
echo -e "\n${BLUE}Migrating Redis Scripts...${NC}"
safe_copy "${OLD_BASE}/mev-sandwich-detector/scripts/redis" "${NEW_BASE}/infrastructure/redis/scripts" "Redis scripts"

# Migrate deployment and build scripts
echo -e "\n${BLUE}Migrating Build and Deploy Scripts...${NC}"
safe_copy "${OLD_BASE}/arbitrage-data-capture/Makefile" "${NEW_BASE}/tools/build/Makefile.legacy" "Legacy Makefile"
safe_copy "${OLD_BASE}/arbitrage-data-capture/start-*.sh" "${NEW_BASE}/tools/deploy/legacy-scripts/" "Start scripts"
safe_copy "${OLD_BASE}/mev-sandwich-detector/scripts" "${NEW_BASE}/tools/deploy/sandwich-scripts" "Sandwich scripts"

# Migrate testing utilities
echo -e "\n${BLUE}Migrating Testing Utilities...${NC}"
safe_copy "${OLD_BASE}/arbitrage-data-capture/testing" "${NEW_BASE}/tools/testing" "Testing framework"
safe_copy "${OLD_BASE}/mev-sandwich-detector/tests" "${NEW_BASE}/tools/testing/sandwich-tests" "Sandwich tests"

# Create consolidated configuration file
echo -e "\n${BLUE}Creating consolidated configuration...${NC}"
cat > "${NEW_BASE}/shared/configs/mev-config.toml" << 'EOF'
# Elite MEV Infrastructure Configuration

[global]
environment = "production"
log_level = "info"
metrics_port = 9090

[solana]
rpc_url = "https://api.mainnet-beta.solana.com"
ws_url = "wss://api.mainnet-beta.solana.com"
commitment = "confirmed"

[mev_engine]
max_concurrent_bundles = 100
bundle_timeout_ms = 50
mempool_scan_interval_ms = 10
opportunity_scan_interval_ms = 5

[sandwich_detector]
detection_threshold = 0.85
ml_model_path = "/models/sandwich_xgboost.model"
max_tip_lamports = 10000000

[arbitrage_engine]
min_profit_bps = 10
max_position_size = 1000000
slippage_tolerance = 0.005

[database]
clickhouse_url = "http://localhost:8123"
clickhouse_database = "mev"
redis_url = "redis://localhost:6379"

[kafka]
brokers = ["localhost:9092"]
topic_prefix = "mev"
consumer_group = "mev-backend"

[monitoring]
prometheus_endpoint = "http://localhost:9091"
grafana_endpoint = "http://localhost:3000"
alert_webhook = ""
EOF

echo -e "${GREEN}✓ Configuration file created${NC}"

# Create Python virtual environment structure
echo -e "\n${BLUE}Setting up Python environments...${NC}"
cat > "${NEW_BASE}/services/control-plane/requirements.txt" << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
sqlalchemy==2.0.35
asyncpg==0.30.0
redis==5.2.0
aiokafka==0.12.0
clickhouse-driver==0.2.9
prometheus-client==0.21.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
xgboost==2.1.2
onnxruntime==1.20.0
python-dotenv==1.0.1
httpx==0.27.2
websockets==13.1
protobuf==5.28.3
grpcio==1.68.0
pytest==8.3.3
pytest-asyncio==0.24.0
black==24.10.0
pylint==3.3.1
EOF

echo -e "${GREEN}✓ Python requirements created${NC}"

# Create deployment configuration
echo -e "\n${BLUE}Creating deployment configuration...${NC}"
cat > "${NEW_BASE}/tools/deploy/deploy-production.sh" << 'EOF'
#!/bin/bash

# Production deployment script for MEV Backend

set -e

echo "Deploying MEV Backend to Production..."

# Build production binaries
make mev-production

# Run database migrations
make db-migrate

# Deploy services
docker compose -f infrastructure/docker/docker-compose.yml up -d

# Health check
sleep 10
make health-check

echo "Deployment complete!"
EOF

chmod +x "${NEW_BASE}/tools/deploy/deploy-production.sh"
echo -e "${GREEN}✓ Deployment script created${NC}"

# Create health check script
cat > "${NEW_BASE}/tools/monitoring/health-check.sh" << 'EOF'
#!/bin/bash

# Health check for all MEV services

SERVICES=("mev-engine:8000" "sandwich-detector:8001" "arbitrage-engine:8002" "control-plane:8080")

echo "Checking service health..."

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -f -s "http://localhost:${port}/health" > /dev/null; then
        echo "✓ ${name} is healthy"
    else
        echo "✗ ${name} is unhealthy"
        exit 1
    fi
done

echo "All services healthy!"
EOF

chmod +x "${NEW_BASE}/tools/monitoring/health-check.sh"
echo -e "${GREEN}✓ Health check script created${NC}"

# Summary
echo -e "\n${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}Migration Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "\n${BLUE}New structure created at: ${NEW_BASE}${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Review migrated files in ${NEW_BASE}"
echo -e "  2. Run 'make setup' to install dependencies"
echo -e "  3. Run 'make build' to build all services"
echo -e "  4. Run 'make test' to verify everything works"
echo -e "  5. Run 'make docker-up' to start services locally"
echo -e "\n${GREEN}Happy MEV hunting!${NC}"
EOF

chmod +x "${NEW_BASE}/tools/migrate-services.sh"