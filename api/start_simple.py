#!/usr/bin/env python3
"""
Simple API server startup with minimal dependencies
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="MEV Infrastructure API",
    description="Defensive-only MEV detection and analysis system",
    version="1.0.0",
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "success": True,
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "message": "API is running (simplified mode)"
    }

# Stub endpoints for testing
@app.post("/auth/login")
async def login():
    return {"access_token": "test_token", "token_type": "bearer"}

@app.post("/clickhouse/query")
async def query():
    return {"success": True, "data": []}

@app.post("/datasets/export")
async def export():
    return {"success": True, "job_id": "test_job"}

@app.post("/training/train")
async def train():
    return {"success": True, "job_id": "test_training"}

@app.post("/control/kill-switch")
async def kill_switch():
    return {"success": True, "status": "inactive"}

@app.get("/control/audit-log")
async def audit_log():
    return {"success": True, "events": []}

@app.websocket("/realtime/ws")
async def websocket_endpoint(websocket):
    await websocket.accept()
    await websocket.send_json({"type": "connected"})
    await websocket.close()

if __name__ == "__main__":
    print("🚀 Starting simplified MEV API Server...")
    print("✅ Health endpoint: http://localhost:8000/health")
    print("📚 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )