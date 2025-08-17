# Ultra-optimized Rust builder for MEV backend
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml ./
COPY shared-types/Cargo.toml ./shared-types/
COPY jito-engine/Cargo.toml ./jito-engine/
COPY quic-service/Cargo.toml ./quic-service/
COPY dex-parser/Cargo.toml ./dex-parser/
COPY arbitrage-engine/Cargo.toml ./arbitrage-engine/
COPY optimizer/Cargo.toml ./optimizer/
COPY metrics-collector/Cargo.toml ./metrics-collector/
COPY alert-system/Cargo.toml ./alert-system/
COPY main-service/Cargo.toml ./main-service/

# Create dummy main files to cache dependencies
RUN mkdir -p shared-types/src jito-engine/src quic-service/src dex-parser/src \
    arbitrage-engine/src optimizer/src metrics-collector/src alert-system/src main-service/src
RUN echo "fn main() {}" > main-service/src/main.rs
RUN touch shared-types/src/lib.rs jito-engine/src/lib.rs quic-service/src/lib.rs \
    dex-parser/src/lib.rs arbitrage-engine/src/lib.rs optimizer/src/lib.rs \
    metrics-collector/src/lib.rs alert-system/src/lib.rs

# Build dependencies
RUN cargo build --release --workspace

# Copy actual source code
COPY . .

# Touch source files to ensure rebuild
RUN touch shared-types/src/lib.rs jito-engine/src/lib.rs quic-service/src/lib.rs \
    dex-parser/src/lib.rs arbitrage-engine/src/lib.rs optimizer/src/lib.rs \
    metrics-collector/src/lib.rs alert-system/src/lib.rs main-service/src/main.rs

# Build with maximum optimizations
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
RUN cargo build --release --bin solana-mev-backend

# Production stage - minimal runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mev && \
    mkdir -p /app /keys /config && \
    chown -R mev:mev /app /keys /config

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/solana-mev-backend /app/

# Set capabilities for network optimization
RUN setcap cap_net_admin,cap_net_raw,cap_sys_nice+eip /app/solana-mev-backend

USER mev

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with optimal settings
CMD ["./solana-mev-backend", "--config", "/config/config.toml"]