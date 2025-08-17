#!/usr/bin/env python3
"""
Control Plane API Service
Binds to 0.0.0.0 for network access
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
from datetime import datetime

app = FastAPI(
    title="Solana MEV Control Plane",
    description="Control plane for MEV infrastructure",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "control-plane",
        "version": "1.0.0"
    }

# API root
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Solana MEV Control Plane",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "metrics": "/api/metrics",
            "opportunities": "/api/opportunities",
            "strategies": "/api/strategies"
        }
    }

# MEV Opportunities
class Opportunity(BaseModel):
    id: str
    type: str
    profit: float
    confidence: float
    timestamp: datetime
    pools: List[str]
    
@app.get("/api/opportunities")
async def get_opportunities(
    type: Optional[str] = None,
    min_profit: Optional[float] = None,
    limit: int = 100
):
    """Get current MEV opportunities"""
    # Mock data for testing
    opportunities = [
        {
            "id": "opp_001",
            "type": "arbitrage",
            "profit": 1.5,
            "confidence": 0.92,
            "timestamp": datetime.utcnow().isoformat(),
            "pools": ["USDC/SOL", "SOL/USDT"],
            "net_profit": 1.45
        },
        {
            "id": "opp_002",
            "type": "sandwich",
            "profit": 0.8,
            "confidence": 0.88,
            "timestamp": datetime.utcnow().isoformat(),
            "pools": ["RAY/SOL"],
            "net_profit": 0.75
        }
    ]
    
    # Filter by type if specified
    if type:
        opportunities = [o for o in opportunities if o["type"] == type]
    
    # Filter by minimum profit
    if min_profit:
        opportunities = [o for o in opportunities if o["profit"] >= min_profit]
    
    return {
        "data": opportunities[:limit],
        "total": len(opportunities),
        "timestamp": datetime.utcnow().isoformat()
    }

# Strategy Management
@app.get("/api/strategies")
async def get_strategies():
    """Get available strategies"""
    return {
        "strategies": [
            {
                "id": "arb_v2",
                "name": "Arbitrage V2",
                "type": "arbitrage",
                "status": "active",
                "performance": {
                    "success_rate": 0.943,
                    "total_profit": 1523.45,
                    "executions": 1240
                }
            },
            {
                "id": "sandwich_v1",
                "name": "Sandwich Detector",
                "type": "sandwich",
                "status": "active",
                "performance": {
                    "success_rate": 0.923,
                    "total_profit": 876.32,
                    "executions": 567
                }
            }
        ]
    }

# Metrics endpoint
@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "system": {
            "uptime": 86400,
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "active_connections": 12
        },
        "mev": {
            "opportunities_detected": 5234,
            "opportunities_executed": 4892,
            "success_rate": 0.935,
            "total_profit": 15234.56
        },
        "network": {
            "rpc_latency_ms": 12.3,
            "ws_connections": 8,
            "packet_rate": 15000
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "trading": {
            "enabled": True,
            "max_position_size": 1000,
            "default_slippage": 0.005
        },
        "strategies": {
            "arbitrage": {"enabled": True, "min_profit": 0.5},
            "sandwich": {"enabled": True, "max_priority_fee": 0.01}
        },
        "network": {
            "rpc_url": os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            "ws_url": os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
        }
    }

@app.post("/api/config")
async def update_config(config: Dict[str, Any]):
    """Update configuration"""
    # In production, this would update actual config
    return {
        "status": "updated",
        "config": config,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    # Configuration for network access
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"🚀 Starting Control Plane API on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔗 Allow remote access: http://<your-ip>:{port}")
    
    # Run with uvicorn, binding to all interfaces
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload for production
        access_log=True,
        log_level="info"
    )