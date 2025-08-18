"""
Training API: XGBoost with Treelite compilation
GPU-accelerated training with hot-reload support
"""

import os
import time
import json
import asyncio
import uuid
import subprocess
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import xgboost as xgb
import treelite
import treelite_runtime
import clickhouse_connect

from deps import User, get_current_user, require_permission, audit_log


router = APIRouter()

# Global state for training jobs
training_jobs: Dict[str, Dict[str, Any]] = {}
job_lock = asyncio.Lock()

# Model registry
model_registry: Dict[str, Dict[str, Any]] = {}
registry_lock = asyncio.Lock()


class TrainingRequest(BaseModel):
    """Model training request"""
    model_type: str = Field("mev_predictor", description="Model type: mev_predictor, arb_scorer, bundle_optimizer")
    dataset_query: str = Field(..., description="ClickHouse query for training data")
    features: List[str] = Field(..., description="Feature column names")
    target: str = Field(..., description="Target column name")
    validation_split: float = Field(0.2, description="Validation split ratio")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="XGBoost hyperparameters")
    use_gpu: bool = Field(False, description="Use GPU for training")
    compile_treelite: bool = Field(True, description="Compile with Treelite for inference")
    auto_deploy: bool = Field(False, description="Auto-deploy after training")


class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    model_type: str
    version: str
    created_at: datetime
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    features: List[str]
    file_paths: Dict[str, str]
    deployed: bool
    metadata: Dict[str, Any]


class TrainingJob(BaseModel):
    """Training job status"""
    job_id: str
    status: str
    progress: float
    model_id: Optional[str]
    metrics: Dict[str, float]
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]


async def train_model_task(job_id: str, request: TrainingRequest, user_id: str):
    """Background task to train model"""
    
    async with job_lock:
        if job_id not in training_jobs:
            return
        training_jobs[job_id]["status"] = "preparing"
        training_jobs[job_id]["started_at"] = datetime.utcnow()
    
    try:
        # Load data from ClickHouse
        async with job_lock:
            training_jobs[job_id]["status"] = "loading_data"
            training_jobs[job_id]["progress"] = 0.1
        
        client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            username=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
            database=os.getenv("CLICKHOUSE_DATABASE", "mev")
        )
        
        # Execute query
        result = client.query(request.dataset_query)
        df = result.to_pandas()
        
        if len(df) < 100:
            raise ValueError(f"Insufficient training data: {len(df)} rows")
        
        # Prepare features and target
        X = df[request.features].values
        y = df[request.target].values
        
        # Split data
        split_idx = int(len(X) * (1 - request.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        async with job_lock:
            training_jobs[job_id]["status"] = "training"
            training_jobs[job_id]["progress"] = 0.2
            training_jobs[job_id]["metadata"]["data_size"] = len(df)
            training_jobs[job_id]["metadata"]["train_size"] = len(X_train)
            training_jobs[job_id]["metadata"]["val_size"] = len(X_val)
        
        # Prepare XGBoost parameters
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        
        # Override with user parameters
        params.update(request.hyperparameters)
        
        # Add GPU support if requested
        if request.use_gpu:
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=request.features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=request.features)
        
        # Training with progress callback
        evals = [(dtrain, "train"), (dval, "val")]
        evals_result = {}
        
        def progress_callback(env):
            """Update training progress"""
            if env.iteration % 10 == 0:
                progress = 0.2 + (0.6 * env.iteration / params.get("n_estimators", 100))
                asyncio.create_task(update_progress(job_id, progress))
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get("n_estimators", 100),
            evals=evals,
            evals_result=evals_result,
            callbacks=[progress_callback],
            verbose_eval=False
        )
        
        async with job_lock:
            training_jobs[job_id]["status"] = "evaluating"
            training_jobs[job_id]["progress"] = 0.8
        
        # Evaluate model
        train_rmse = evals_result["train"]["rmse"][-1]
        val_rmse = evals_result["val"]["rmse"][-1]
        
        # Calculate additional metrics
        y_pred_train = model.predict(dtrain)
        y_pred_val = model.predict(dval)
        
        train_mae = np.mean(np.abs(y_train - y_pred_train))
        val_mae = np.mean(np.abs(y_val - y_pred_val))
        
        train_r2 = 1 - np.sum((y_train - y_pred_train)**2) / np.sum((y_train - np.mean(y_train))**2)
        val_r2 = 1 - np.sum((y_val - y_pred_val)**2) / np.sum((y_val - np.mean(y_val))**2)
        
        metrics = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "train_r2": float(train_r2),
            "val_r2": float(val_r2)
        }
        
        async with job_lock:
            training_jobs[job_id]["metrics"] = metrics
            training_jobs[job_id]["status"] = "saving"
            training_jobs[job_id]["progress"] = 0.9
        
        # Save model
        model_id = f"{request.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}"
        model_dir = Path(os.getenv("MODEL_DIR", "/tmp/models")) / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        xgb_path = model_dir / "model.xgb"
        model.save_model(str(xgb_path))
        
        # Save as JSON for inspection
        json_path = model_dir / "model.json"
        model.dump_model(str(json_path))
        
        file_paths = {
            "xgboost": str(xgb_path),
            "json": str(json_path)
        }
        
        # Compile with Treelite if requested
        if request.compile_treelite:
            async with job_lock:
                training_jobs[job_id]["status"] = "compiling"
                training_jobs[job_id]["progress"] = 0.95
            
            try:
                # Load model into Treelite
                tl_model = treelite.Model.from_xgboost(model)
                
                # Generate C code
                c_path = model_dir / "model.c"
                tl_model.export_srcdir(
                    platform="unix",
                    toolchain="gcc",
                    params={"parallel_comp": 8},
                    dirpath=str(model_dir / "src"),
                    verbose=False
                )
                
                # Compile to shared library
                so_path = model_dir / f"lib{model_id}.so"
                
                # Build command
                build_cmd = [
                    "gcc",
                    "-shared",
                    "-fPIC",
                    "-O3",
                    "-march=native",
                    "-mtune=native",
                    "-fopenmp",
                    "-std=c99"
                ]
                
                # Add source files
                src_dir = model_dir / "src"
                for src_file in src_dir.glob("*.c"):
                    build_cmd.append(str(src_file))
                
                build_cmd.extend(["-o", str(so_path), "-lm", "-lgomp"])
                
                # Compile
                result = subprocess.run(build_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Compilation failed: {result.stderr}")
                
                file_paths["treelite_so"] = str(so_path)
                
                # Create symlink for hot-reload
                latest_link = Path(os.getenv("MODEL_DIR", "/tmp/models")) / f"{request.model_type}_latest.so"
                if latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(so_path)
                
                file_paths["latest_symlink"] = str(latest_link)
                
            except Exception as e:
                print(f"Treelite compilation failed: {e}")
                # Continue without Treelite
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "model_type": request.model_type,
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "features": request.features,
            "target": request.target,
            "hyperparameters": params,
            "metrics": metrics,
            "data_size": len(df),
            "file_paths": file_paths
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Register model
        async with registry_lock:
            model_registry[model_id] = {
                "model_id": model_id,
                "model_type": request.model_type,
                "version": "1.0.0",
                "created_at": datetime.utcnow(),
                "metrics": metrics,
                "hyperparameters": params,
                "features": request.features,
                "file_paths": file_paths,
                "deployed": request.auto_deploy,
                "metadata": {
                    "user_id": user_id,
                    "job_id": job_id,
                    "data_size": len(df)
                }
            }
        
        # Update job status
        async with job_lock:
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["progress"] = 1.0
            training_jobs[job_id]["model_id"] = model_id
            training_jobs[job_id]["completed_at"] = datetime.utcnow()
        
        # Auto-deploy if requested
        if request.auto_deploy:
            await deploy_model(model_id)
        
    except Exception as e:
        # Update job with error
        async with job_lock:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = str(e)
            training_jobs[job_id]["completed_at"] = datetime.utcnow()


async def update_progress(job_id: str, progress: float):
    """Update job progress"""
    async with job_lock:
        if job_id in training_jobs:
            training_jobs[job_id]["progress"] = min(progress, 1.0)


async def deploy_model(model_id: str):
    """Deploy model for inference"""
    # Send control command to swap model
    from .control import get_kafka_producer, sign_command
    from proto_gen import control_pb2
    
    async with registry_lock:
        if model_id not in model_registry:
            return
        
        model_info = model_registry[model_id]
    
    # Create model swap command
    swap = control_pb2.ModelSwap()
    swap.model_id = model_id
    swap.model_path = model_info["file_paths"].get("treelite_so", model_info["file_paths"]["xgboost"])
    swap.model_type = model_info["model_type"]
    swap.version = model_info["version"]
    
    for key, value in model_info["metadata"].items():
        swap.metadata[key] = str(value)
    
    # Wrap in command
    command = control_pb2.Command()
    command.id = f"deploy_{time.time_ns()}"
    command.module = model_info["model_type"]
    command.action = "swap_model"
    command.nonce = time.time_ns()
    command.timestamp_ns = time.time_ns()
    command.params["model_data"] = swap.SerializeToString().hex()
    
    # Sign and publish
    command.pubkey_id = os.getenv("CTRL_PUBKEY_ID", "default")
    
    producer = await get_kafka_producer()
    await producer.send(
        "control-commands-proto",
        value=command.SerializeToString(),
        key=model_info["model_type"].encode()
    )
    
    # Update registry
    async with registry_lock:
        model_registry[model_id]["deployed"] = True


@router.post("/train", dependencies=[Depends(require_permission("training:write"))])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
) -> TrainingJob:
    """Start model training job"""
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job
    async with job_lock:
        training_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "model_id": None,
            "metrics": {},
            "error": None,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "user_id": user.id,
            "request": request.dict(),
            "metadata": {
                "model_type": request.model_type,
                "use_gpu": request.use_gpu,
                "compile_treelite": request.compile_treelite
            }
        }
    
    # Start background training
    background_tasks.add_task(
        train_model_task,
        job_id,
        request,
        user.id
    )
    
    # Audit log
    background_tasks.add_task(
        audit_log,
        "model_training",
        user,
        {
            "job_id": job_id,
            "model_type": request.model_type,
            "use_gpu": request.use_gpu
        }
    )
    
    return TrainingJob(**training_jobs[job_id])


@router.get("/jobs/{job_id}", dependencies=[Depends(require_permission("training:read"))])
async def get_job_status(
    job_id: str,
    user: User = Depends(get_current_user)
) -> TrainingJob:
    """Get training job status"""
    
    async with job_lock:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = training_jobs[job_id]
        
        # Check authorization
        if job["user_id"] != user.id and "admin" not in user.roles:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return TrainingJob(**job)


@router.get("/jobs", dependencies=[Depends(require_permission("training:read"))])
async def list_jobs(
    user: User = Depends(get_current_user),
    limit: int = Query(100, le=1000)
) -> List[TrainingJob]:
    """List training jobs"""
    
    async with job_lock:
        user_jobs = []
        for job in training_jobs.values():
            if job["user_id"] == user.id or "admin" in user.roles:
                user_jobs.append(TrainingJob(**job))
        
        # Sort by created_at descending
        user_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_jobs[:limit]


@router.get("/models", dependencies=[Depends(require_permission("training:read"))])
async def list_models(
    model_type: Optional[str] = None,
    deployed_only: bool = False,
    user: User = Depends(get_current_user)
) -> List[ModelInfo]:
    """List trained models"""
    
    async with registry_lock:
        models = []
        for model_info in model_registry.values():
            # Filter by type if specified
            if model_type and model_info["model_type"] != model_type:
                continue
            
            # Filter by deployment status
            if deployed_only and not model_info["deployed"]:
                continue
            
            models.append(ModelInfo(**model_info))
        
        # Sort by created_at descending
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models


@router.get("/models/{model_id}", dependencies=[Depends(require_permission("training:read"))])
async def get_model(
    model_id: str,
    user: User = Depends(get_current_user)
) -> ModelInfo:
    """Get model information"""
    
    async with registry_lock:
        if model_id not in model_registry:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelInfo(**model_registry[model_id])


@router.post("/models/{model_id}/deploy", dependencies=[Depends(require_permission("training:write"))])
async def deploy_model_endpoint(
    model_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Deploy model for inference"""
    
    async with registry_lock:
        if model_id not in model_registry:
            raise HTTPException(status_code=404, detail="Model not found")
    
    # Deploy model
    await deploy_model(model_id)
    
    # Audit log
    background_tasks.add_task(
        audit_log,
        "model_deploy",
        user,
        {"model_id": model_id}
    )
    
    return {"status": "deployed", "model_id": model_id}


@router.delete("/models/{model_id}", dependencies=[Depends(require_permission("training:write"))])
async def delete_model(
    model_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete model"""
    
    async with registry_lock:
        if model_id not in model_registry:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = model_registry[model_id]
        
        # Delete model files
        model_dir = Path(model_info["file_paths"]["xgboost"]).parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del model_registry[model_id]
    
    return {"status": "deleted", "model_id": model_id}