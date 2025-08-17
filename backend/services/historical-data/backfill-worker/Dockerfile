# Multi-stage build for Node.js backfill worker
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json tsconfig.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm ci --only=development

# Copy source code
COPY src ./src

# Build TypeScript
RUN npm run build

# Runtime stage
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    tini

# Create non-root user
RUN addgroup -g 1000 solana && \
    adduser -D -u 1000 -G solana solana

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production && \
    npm cache clean --force

# Copy built application
COPY --from=builder --chown=solana:solana /app/dist ./dist

USER solana

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9091/health || exit 1

EXPOSE 9091

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["node", "dist/index.js"]