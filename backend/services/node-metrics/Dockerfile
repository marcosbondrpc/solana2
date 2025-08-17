# Build stage
FROM rust:1.75 as builder

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY services/node-metrics/Cargo.toml services/node-metrics/

# Create dummy main to cache dependencies
RUN mkdir -p services/node-metrics/src && \
    echo "fn main() {}" > services/node-metrics/src/main.rs

# Build dependencies
RUN cargo build --release --package node-metrics

# Copy actual source code
COPY services/node-metrics/src services/node-metrics/src

# Touch main.rs to trigger rebuild with actual source
RUN touch services/node-metrics/src/main.rs

# Build the actual application
RUN cargo build --release --package node-metrics

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mev && \
    mkdir -p /app && \
    chown -R mev:mev /app

USER mev
WORKDIR /app

# Copy binary from builder
COPY --from=builder --chown=mev:mev /app/target/release/node-metrics /app/node-metrics

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9090/metrics || exit 1

# Expose ports
EXPOSE 8081 9090

# Run the service
CMD ["./node-metrics"]