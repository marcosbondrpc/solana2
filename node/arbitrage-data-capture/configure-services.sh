#!/bin/bash

#########################################################################
# Configure Services for Arbitrage Data Capture
# Sets up Kafka, Redis, and ClickHouse for optimal performance
#########################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          CONFIGURING ARBITRAGE DATA CAPTURE SERVICES              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. Configure ClickHouse
configure_clickhouse() {
    echo -e "${YELLOW}[1/4] Configuring ClickHouse...${NC}"
    
    # Apply schema
    if clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
        echo -e "${YELLOW}Creating ClickHouse database and tables...${NC}"
        clickhouse-client < "$SCRIPT_DIR/clickhouse-setup.sql"
        echo -e "${GREEN}✓ ClickHouse configured${NC}"
    else
        echo -e "${RED}✗ ClickHouse is not running. Please start it first.${NC}"
        echo -e "${YELLOW}Run: sudo systemctl start clickhouse-server${NC}"
        exit 1
    fi
}

# 2. Configure Kafka
configure_kafka() {
    echo -e "${YELLOW}[2/4] Configuring Kafka...${NC}"
    
    KAFKA_DIR="/opt/kafka"
    
    if [ ! -d "$KAFKA_DIR" ]; then
        echo -e "${RED}✗ Kafka not found. Please run install-dependencies.sh first.${NC}"
        exit 1
    fi
    
    # Create Kafka configuration
    cat > "$SCRIPT_DIR/kafka-server.properties" << 'EOF'
# Kafka Server Configuration for Solana Arbitrage Data
broker.id=0
listeners=PLAINTEXT://localhost:9092
advertised.listeners=PLAINTEXT://localhost:9092
log.dirs=/var/kafka-logs
num.network.threads=8
num.io.threads=8
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# Log retention (7 days)
log.retention.hours=168
log.segment.bytes=1073741824
log.retention.check.interval.ms=300000

# Zookeeper
zookeeper.connect=localhost:2181
zookeeper.connection.timeout.ms=18000

# Performance tuning
num.replica.fetchers=4
replica.fetch.max.bytes=1048576
replica.socket.receive.buffer.bytes=65536

# Compression
compression.type=lz4

# Group coordinator
group.initial.rebalance.delay.ms=0
EOF

    # Create Zookeeper configuration
    cat > "$SCRIPT_DIR/zookeeper.properties" << 'EOF'
dataDir=/var/zookeeper
clientPort=2181
maxClientCnxns=0
admin.enableServer=false
tickTime=2000
initLimit=10
syncLimit=5
EOF

    # Create systemd service for Zookeeper
    sudo tee /etc/systemd/system/zookeeper.service > /dev/null << EOF
[Unit]
Description=Apache Zookeeper
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=$KAFKA_DIR/bin/zookeeper-server-start.sh $SCRIPT_DIR/zookeeper.properties
ExecStop=$KAFKA_DIR/bin/zookeeper-server-stop.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

    # Create systemd service for Kafka
    sudo tee /etc/systemd/system/kafka.service > /dev/null << EOF
[Unit]
Description=Apache Kafka
After=zookeeper.service
Requires=zookeeper.service

[Service]
Type=simple
User=$USER
ExecStart=$KAFKA_DIR/bin/kafka-server-start.sh $SCRIPT_DIR/kafka-server.properties
ExecStop=$KAFKA_DIR/bin/kafka-server-stop.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

    # Create Kafka topics
    cat > "$SCRIPT_DIR/create-topics.sh" << 'EOF'
#!/bin/bash
KAFKA_DIR="/opt/kafka"

# Create topics with optimal partitioning
$KAFKA_DIR/bin/kafka-topics.sh --create --topic solana-transactions --bootstrap-server localhost:9092 --partitions 8 --replication-factor 1 --config retention.ms=604800000 --config compression.type=lz4
$KAFKA_DIR/bin/kafka-topics.sh --create --topic arbitrage-opportunities --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1 --config retention.ms=86400000
$KAFKA_DIR/bin/kafka-topics.sh --create --topic dex-pool-states --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1 --config retention.ms=259200000
EOF
    chmod +x "$SCRIPT_DIR/create-topics.sh"
    
    sudo systemctl daemon-reload
    echo -e "${GREEN}✓ Kafka configured${NC}"
}

# 3. Configure Redis
configure_redis() {
    echo -e "${YELLOW}[3/4] Configuring Redis...${NC}"
    
    # Create Redis configuration
    cat > "$SCRIPT_DIR/redis.conf" << 'EOF'
# Redis Configuration for Arbitrage Data Cache

# Network
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300

# Memory Management
maxmemory 4gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence (AOF for durability)
appendonly yes
appendfilename "arbitrage-cache.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Performance
databases 16
rdbcompression yes
rdbchecksum yes
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Lua scripting
lua-time-limit 5000

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Advanced config
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100
activerehashing yes
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
hz 10
dynamic-hz yes
EOF

    # Update Redis systemd service to use our config
    sudo tee /etc/systemd/system/redis-custom.service > /dev/null << EOF
[Unit]
Description=Redis In-Memory Data Store for Arbitrage Cache
After=network.target

[Service]
Type=notify
ExecStart=/usr/bin/redis-server $SCRIPT_DIR/redis.conf
ExecStop=/usr/bin/redis-cli shutdown
TimeoutStopSec=0
Restart=on-failure
User=redis
Group=redis
RuntimeDirectory=redis
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo -e "${GREEN}✓ Redis configured${NC}"
}

# 4. Create monitoring script
create_monitoring() {
    echo -e "${YELLOW}[4/4] Creating monitoring scripts...${NC}"
    
    cat > "$SCRIPT_DIR/monitor-services.sh" << 'EOF'
#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    SERVICE STATUS MONITOR                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

# Check ClickHouse
echo -n "ClickHouse: "
if clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
    COUNT=$(clickhouse-client --query "SELECT count() FROM solana_arbitrage.transactions" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓ Running${NC} (Records: $COUNT)"
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check Kafka
echo -n "Kafka: "
if /opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check Redis
echo -n "Redis: "
if redis-cli ping > /dev/null 2>&1; then
    KEYS=$(redis-cli dbsize | awk '{print $2}')
    echo -e "${GREEN}✓ Running${NC} (Keys: $KEYS)"
else
    echo -e "${RED}✗ Not running${NC}"
fi

echo
echo "Data Pipeline Metrics:"
echo "----------------------"

# Kafka lag
if command -v /opt/kafka/bin/kafka-consumer-groups.sh > /dev/null 2>&1; then
    echo "Kafka Consumer Lag:"
    /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group clickhouse-consumer --describe 2>/dev/null | grep -E "TOPIC|solana" || echo "No consumer groups found"
fi

# ClickHouse compression stats
if clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
    echo -e "\nClickHouse Compression:"
    clickhouse-client --query "
    SELECT 
        table,
        formatReadableSize(sum(data_compressed_bytes)) as compressed,
        formatReadableSize(sum(data_uncompressed_bytes)) as uncompressed,
        round(sum(data_uncompressed_bytes) / sum(data_compressed_bytes), 2) as ratio
    FROM system.parts
    WHERE database = 'solana_arbitrage' AND active
    GROUP BY table
    FORMAT Pretty" 2>/dev/null || echo "No data yet"
fi

# Redis memory usage
if redis-cli ping > /dev/null 2>&1; then
    echo -e "\nRedis Memory:"
    redis-cli info memory | grep -E "used_memory_human|used_memory_peak_human"
fi
EOF
    chmod +x "$SCRIPT_DIR/monitor-services.sh"
    
    echo -e "${GREEN}✓ Monitoring scripts created${NC}"
}

# Main execution
main() {
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then 
        echo -e "${RED}Please do not run as root${NC}"
        exit 1
    fi
    
    # Create necessary directories
    sudo mkdir -p /var/kafka-logs /var/zookeeper /var/log/redis
    sudo chown -R $USER:$USER /var/kafka-logs /var/zookeeper
    
    # Configure services
    configure_clickhouse
    configure_kafka
    configure_redis
    create_monitoring
    
    echo
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ All services configured successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${BLUE}To start services:${NC}"
    echo -e "1. Start Zookeeper: ${YELLOW}sudo systemctl start zookeeper${NC}"
    echo -e "2. Start Kafka: ${YELLOW}sudo systemctl start kafka${NC}"
    echo -e "3. Create topics: ${YELLOW}./create-topics.sh${NC}"
    echo -e "4. Start Redis: ${YELLOW}sudo systemctl start redis-custom${NC}"
    echo -e "5. Monitor services: ${YELLOW}./monitor-services.sh${NC}"
}

main "$@"