#!/bin/bash

#########################################################################
# Arbitrage Data Capture System - Installation Script
# Installs Kafka and Redis for Solana node data pipeline
#########################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     SOLANA ARBITRAGE DATA CAPTURE - DEPENDENCY INSTALLER          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please do not run as root. Use sudo when prompted.${NC}"
   exit 1
fi

# Function to install Kafka
install_kafka() {
    echo -e "${YELLOW}[1/3] Installing Apache Kafka...${NC}"
    
    # Check Java installation
    if ! command -v java &> /dev/null; then
        echo -e "${YELLOW}Installing Java (required for Kafka)...${NC}"
        sudo apt-get update
        sudo apt-get install -y openjdk-11-jdk
    fi
    
    # Download and install Kafka
    KAFKA_VERSION="3.5.1"
    SCALA_VERSION="2.13"
    KAFKA_DIR="/opt/kafka"
    
    if [ ! -d "$KAFKA_DIR" ]; then
        echo -e "${YELLOW}Downloading Kafka ${KAFKA_VERSION}...${NC}"
        cd /tmp
        wget -q "https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
        
        echo -e "${YELLOW}Extracting Kafka...${NC}"
        sudo tar -xzf "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
        sudo mv "kafka_${SCALA_VERSION}-${KAFKA_VERSION}" $KAFKA_DIR
        sudo chown -R $USER:$USER $KAFKA_DIR
        
        # Clean up
        rm "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
        
        # Add Kafka to PATH
        echo "export PATH=\$PATH:${KAFKA_DIR}/bin" >> ~/.bashrc
        
        echo -e "${GREEN}✓ Kafka installed successfully${NC}"
    else
        echo -e "${GREEN}✓ Kafka already installed${NC}"
    fi
}

# Function to install Redis
install_redis() {
    echo -e "${YELLOW}[2/3] Installing Redis...${NC}"
    
    # Update package list
    sudo apt-get update
    
    # Install Redis
    sudo apt-get install -y redis-server redis-tools
    
    # Enable and start Redis
    sudo systemctl enable redis-server
    sudo systemctl start redis-server
    
    # Check Redis status
    if redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Redis installed and running${NC}"
    else
        echo -e "${RED}✗ Redis installation failed${NC}"
        exit 1
    fi
}

# Function to install Rust dependencies
install_rust_deps() {
    echo -e "${YELLOW}[3/3] Installing Rust dependencies...${NC}"
    
    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        echo -e "${YELLOW}Installing Rust...${NC}"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi
    
    # Create Rust project directory
    mkdir -p ~/0solana/node/arbitrage-data-capture/rust-services
    
    echo -e "${GREEN}✓ Rust dependencies ready${NC}"
}

# Main installation
echo -e "${BLUE}Starting installation...${NC}"
echo

# Install Kafka
install_kafka

# Install Redis
install_redis

# Install Rust dependencies
install_rust_deps

# Create necessary directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p ~/0solana/node/arbitrage-data-capture/{config,scripts,data,logs}

echo
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ All dependencies installed successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Source your bashrc: ${YELLOW}source ~/.bashrc${NC}"
echo -e "2. Start Kafka: ${YELLOW}cd /opt/kafka && ./bin/kafka-server-start.sh config/server.properties${NC}"
echo -e "3. Check Redis: ${YELLOW}redis-cli ping${NC}"
echo -e "4. Run configuration script: ${YELLOW}./configure-services.sh${NC}"