#!/bin/bash

# Ultra-high-performance build script for Solana MEV Frontend

set -e

echo "🚀 Starting ultra-high-performance build process..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build environment
BUILD_ENV=${BUILD_ENV:-production}
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc)}

echo -e "${BLUE}Build Configuration:${NC}"
echo "  Environment: $BUILD_ENV"
echo "  Parallel Jobs: $PARALLEL_JOBS"
echo ""

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
npm run clean

# Install dependencies with frozen lockfile
echo -e "${YELLOW}Installing dependencies...${NC}"
npm ci --prefer-offline --no-audit

# Type checking in parallel
echo -e "${YELLOW}Running type checks...${NC}"
npm run type-check

# Linting in parallel
echo -e "${YELLOW}Running linters...${NC}"
npm run lint

# Build packages first (dependencies)
echo -e "${BLUE}Building packages...${NC}"
npm run turbo run build --filter='./packages/*' --concurrency=$PARALLEL_JOBS

# Build apps
echo -e "${BLUE}Building applications...${NC}"
npm run turbo run build --filter='./apps/*' --concurrency=$PARALLEL_JOBS

# Run tests
if [ "$RUN_TESTS" = "true" ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    npm run test
fi

# Bundle analysis
if [ "$ANALYZE" = "true" ]; then
    echo -e "${YELLOW}Generating bundle analysis...${NC}"
    npm run analyze
fi

# Optimize assets
echo -e "${YELLOW}Optimizing assets...${NC}"

# Compress HTML files
find ./apps/*/dist -name "*.html" -exec html-minifier-terser \
    --collapse-whitespace \
    --remove-comments \
    --remove-optional-tags \
    --remove-redundant-attributes \
    --remove-script-type-attributes \
    --remove-tag-whitespace \
    --use-short-doctype \
    --minify-css true \
    --minify-js true \
    -o {} {} \; 2>/dev/null || true

# Optimize images
find ./apps/*/dist -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) -exec \
    imagemin {} --out-dir=$(dirname {}) \; 2>/dev/null || true

# Generate service worker
echo -e "${YELLOW}Generating service worker...${NC}"
workbox generateSW workbox-config.js 2>/dev/null || true

# Calculate build size
echo -e "${BLUE}Build Statistics:${NC}"
for app in ./apps/*/dist; do
    if [ -d "$app" ]; then
        app_name=$(basename $(dirname $app))
        size=$(du -sh $app | cut -f1)
        echo "  $app_name: $size"
    fi
done

# Generate build manifest
echo -e "${YELLOW}Generating build manifest...${NC}"
cat > build-manifest.json << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": "$BUILD_ENV",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "node_version": "$(node -v)",
  "npm_version": "$(npm -v)"
}
EOF

echo -e "${GREEN}✅ Build completed successfully!${NC}"

# Performance metrics
if [ "$SHOW_METRICS" = "true" ]; then
    echo -e "${BLUE}Performance Metrics:${NC}"
    echo "  Total build time: ${SECONDS}s"
    echo "  CPU usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
    echo "  Memory usage: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
fi