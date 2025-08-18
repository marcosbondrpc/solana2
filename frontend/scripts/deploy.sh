#!/bin/bash

# Deployment script for Solana MEV Frontend

set -e

# Configuration
DEPLOY_ENV=${1:-production}
DEPLOY_TARGET=${2:-vercel}

echo "🚀 Deploying Solana MEV Frontend"
echo "Environment: $DEPLOY_ENV"
echo "Target: $DEPLOY_TARGET"

# Build the applications
./scripts/build.sh

case $DEPLOY_TARGET in
  vercel)
    echo "Deploying to Vercel..."
    vercel --prod --confirm
    ;;
    
  cloudflare)
    echo "Deploying to Cloudflare Pages..."
    wrangler pages publish ./apps/dashboard/dist --project-name=solana-mev-dashboard
    ;;
    
  docker)
    echo "Building Docker images..."
    docker build -t solana-mev-frontend:latest .
    docker tag solana-mev-frontend:latest solana-mev-frontend:$DEPLOY_ENV
    
    if [ "$PUSH_TO_REGISTRY" = "true" ]; then
      docker push solana-mev-frontend:latest
      docker push solana-mev-frontend:$DEPLOY_ENV
    fi
    ;;
    
  k8s)
    echo "Deploying to Kubernetes..."
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml
    
    # Wait for deployment
    kubectl rollout status deployment/mev-dashboard -n solana-mev
    ;;
    
  aws)
    echo "Deploying to AWS S3 + CloudFront..."
    aws s3 sync ./apps/dashboard/dist s3://solana-mev-dashboard --delete
    aws cloudfront create-invalidation --distribution-id $CLOUDFRONT_DIST_ID --paths "/*"
    ;;
    
  *)
    echo "Unknown deployment target: $DEPLOY_TARGET"
    exit 1
    ;;
esac

echo "✅ Deployment completed successfully!"