# Build stage
FROM rust:1.75 as builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source
COPY src ./src

# Build for release
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/mission-control /app/mission-control

# Environment variables
ENV RUST_LOG=info
ENV REDIS_URL=redis://redis:6379
ENV RPC_ENDPOINTS=https://api.mainnet-beta.solana.com
ENV TPU_ENDPOINT=127.0.0.1:8003
ENV VALIDATOR_STAKE=1000000000000

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the binary
CMD ["./mission-control"]