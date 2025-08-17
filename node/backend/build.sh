#!/bin/bash

set -e

echo "Building ultra-high-performance Solana MEV backend..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Rust is not installed${NC}"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check Rust version
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo -e "${GREEN}Using Rust version: $RUST_VERSION${NC}"

# Set build flags for maximum performance
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
export CARGO_PROFILE_RELEASE_LTO="fat"
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
export CARGO_PROFILE_RELEASE_OPT_LEVEL=3

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
cargo clean

# Build all workspace members
echo -e "${YELLOW}Building workspace with maximum optimizations...${NC}"
cargo build --release --workspace

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
cargo test --release --workspace

# Build Docker image if Docker is available
if command -v docker &> /dev/null; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t solana-mev-backend:latest .
    echo -e "${GREEN}Docker image built successfully${NC}"
else
    echo -e "${YELLOW}Docker not found, skipping Docker build${NC}"
fi

# Create directories for runtime
mkdir -p keys config logs

# Generate example Jito auth keypair if it doesn't exist
if [ ! -f "keys/jito-auth.json" ]; then
    echo -e "${YELLOW}Generating example Jito auth keypair...${NC}"
    if command -v solana-keygen &> /dev/null; then
        solana-keygen new --no-passphrase --outfile keys/jito-auth.json
    else
        echo -e "${YELLOW}Solana CLI not found, please generate keypair manually${NC}"
    fi
fi

# Show binary location
BINARY_PATH="$(pwd)/target/release/solana-mev-backend"
if [ -f "$BINARY_PATH" ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}Binary location: $BINARY_PATH${NC}"
    echo -e "${GREEN}Binary size: $(du -h $BINARY_PATH | cut -f1)${NC}"
    
    # Show performance stats
    echo -e "\n${YELLOW}Performance optimizations applied:${NC}"
    echo "  - Native CPU instructions (target-cpu=native)"
    echo "  - Link-time optimization (LTO=fat)"
    echo "  - Single codegen unit for maximum inlining"
    echo "  - Optimization level 3"
    echo "  - Strip symbols for smaller binary"
    
    echo -e "\n${YELLOW}To run the backend:${NC}"
    echo "  ./target/release/solana-mev-backend --config config.toml"
    
    echo -e "\n${YELLOW}To run with Docker:${NC}"
    echo "  docker-compose up -d mev-backend"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi