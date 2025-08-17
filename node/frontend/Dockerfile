# Multi-stage Dockerfile for ultra-optimized production builds

# Stage 1: Dependencies
FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat python3 make g++
WORKDIR /app

# Copy package files
COPY package.json package-lock.json turbo.json ./
COPY apps/*/package.json apps/*/
COPY packages/*/package.json packages/*/

# Install dependencies with frozen lockfile
RUN npm ci --only=production

# Stage 2: Builder
FROM node:20-alpine AS builder
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build all applications
ENV NODE_ENV=production
ENV BUILD_ENV=production
RUN npm run build

# Stage 3: Runtime - Dashboard
FROM nginx:alpine AS dashboard
WORKDIR /usr/share/nginx/html

# Copy custom nginx config
COPY --from=builder /app/nginx.conf /etc/nginx/nginx.conf

# Copy built dashboard app
COPY --from=builder /app/apps/dashboard/dist .

# Add security headers
RUN echo 'add_header X-Frame-Options "SAMEORIGIN" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header X-Content-Type-Options "nosniff" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header X-XSS-Protection "1; mode=block" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header Referrer-Policy "strict-origin-when-cross-origin" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header Content-Security-Policy "default-src '\''self'\''; script-src '\''self'\'' '\''unsafe-inline'\''; style-src '\''self'\'' '\''unsafe-inline'\''; img-src '\''self'\'' data: https:; font-src '\''self'\'' data:; connect-src '\''self'\'' wss: https:;" always;' >> /etc/nginx/conf.d/security.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# Stage 4: Runtime - Operator
FROM nginx:alpine AS operator
WORKDIR /usr/share/nginx/html

COPY --from=builder /app/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/apps/operator/dist .

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# Stage 5: Runtime - Analytics
FROM nginx:alpine AS analytics
WORKDIR /usr/share/nginx/html

COPY --from=builder /app/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/apps/analytics/dist .

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]