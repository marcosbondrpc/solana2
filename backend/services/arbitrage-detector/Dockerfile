# Multi-stage build for optimized arbitrage detector
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build with maximum optimizations
ENV CARGO_TARGET_DIR=/build/target
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash arbitrage

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/arbitrage-detector /app/arbitrage-detector

# Set ownership
RUN chown -R arbitrage:arbitrage /app

USER arbitrage

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${METRICS_PORT:-9092}/health || exit 1

# Default environment variables
ENV RUST_LOG=info \
    RUST_BACKTRACE=1

EXPOSE 8090 9092

ENTRYPOINT ["/app/arbitrage-detector"]