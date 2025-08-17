# Build stage
FROM rust:1.75 as builder

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY services/data-scrapper/Cargo.toml services/data-scrapper/

# Create dummy main to cache dependencies
RUN mkdir -p services/data-scrapper/src && \
    echo "fn main() {}" > services/data-scrapper/src/main.rs

# Build dependencies
RUN cargo build --release --package data-scrapper

# Copy actual source code
COPY services/data-scrapper/src services/data-scrapper/src

# Touch main.rs to trigger rebuild with actual source
RUN touch services/data-scrapper/src/main.rs

# Build the actual application
RUN cargo build --release --package data-scrapper

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and data directories
RUN useradd -m -u 1000 mev && \
    mkdir -p /app /data/scrapper /data/datasets /data/models && \
    chown -R mev:mev /app /data

USER mev
WORKDIR /app

# Copy binary from builder
COPY --from=builder --chown=mev:mev /app/target/release/data-scrapper /app/data-scrapper

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

# Expose API port
EXPOSE 8082

# Mount point for data persistence
VOLUME ["/data"]

# Run the service
CMD ["./data-scrapper"]