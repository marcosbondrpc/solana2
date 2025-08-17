"""
ML model training endpoints
"""

import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from models.schemas import (
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    ModelDeployRequest,
    ModelStatus,
    DeploymentMode,
    UserRole
)
from security.auth import require_role, get_current_user, TokenData
from security.policy import Permission
from services.clickhouse_client import get_clickhouse_pool

router = APIRouter()


class TrainingService:
    """ML training service manager"""
    
    def __init__(self):
        self.training_jobs: Dict[str, Dict] = {}
        self.models: Dict[str, ModelInfo] = {}
        self.deployments: Dict[str, Dict] = {}
    
    async def start_training(
        self,
        job_id: str,
        request: TrainingRequest,
        user_id: str
    ):
        """Start model training job"""
        
        # Initialize job
        self.training_jobs[job_id] = {
            "job_id": job_id,
            "model_id": f"model_{job_id[:8]}",
            "status": ModelStatus.TRAINING,
            "started_at": datetime.now(),
            "user_id": user_id,
            "request": request.dict(),
            "progress": 0.0
        }
        
        try:
            # Simulate training process (in production, this would call actual ML pipeline)
            import asyncio
            
            # Fetch training data
            pool = await get_clickhouse_pool()
            data, _ = await pool.execute_query(
                request.dataset_query,
                use_cache=False
            )
            
            if not data:
                raise ValueError("No training data found")
            
            # Update progress
            self.training_jobs[job_id]["progress"] = 20.0
            
            # Simulate feature engineering
            await asyncio.sleep(2)
            self.training_jobs[job_id]["progress"] = 40.0
            
            # Simulate model training
            await asyncio.sleep(3)
            self.training_jobs[job_id]["progress"] = 60.0
            
            # Simulate validation
            await asyncio.sleep(2)
            self.training_jobs[job_id]["progress"] = 80.0
            
            # Simulate Treelite compilation if requested
            if request.compile_treelite:
                await asyncio.sleep(1)
            
            # Generate mock metrics
            metrics = {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.915,
                "auc_roc": 0.96,
                "latency_p99_us": 87.5
            }
            
            # Create model entry
            model_info = ModelInfo(
                model_id=self.training_jobs[job_id]["model_id"],
                version="1.0.0",
                model_type=request.model_type,
                status=ModelStatus.READY,
                created_at=datetime.now(),
                trained_by=user_id,
                accuracy=metrics["accuracy"],
                latency_p99_us=metrics["latency_p99_us"],
                feature_importance={
                    feature: 0.1 + (i * 0.05)
                    for i, feature in enumerate(request.features[:10])
                },
                metadata={
                    "training_rows": len(data),
                    "features": request.features,
                    "target": request.target,
                    "hyperparameters": request.hyperparameters or {},
                    "metrics": metrics
                }
            )
            
            self.models[model_info.model_id] = model_info
            
            # Update job status
            self.training_jobs[job_id]["status"] = ModelStatus.READY
            self.training_jobs[job_id]["progress"] = 100.0
            self.training_jobs[job_id]["metrics"] = metrics
            self.training_jobs[job_id]["completed_at"] = datetime.now()
            
        except Exception as e:
            self.training_jobs[job_id]["status"] = ModelStatus.FAILED
            self.training_jobs[job_id]["error"] = str(e)
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get training job status"""
        return self.training_jobs.get(job_id)
    
    def list_models(self, user_id: Optional[str] = None) -> List[ModelInfo]:
        """List trained models"""
        models = list(self.models.values())
        
        if user_id:
            models = [m for m in models if m.trained_by == user_id]
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info"""
        return self.models.get(model_id)
    
    async def deploy_model(
        self,
        model_id: str,
        request: ModelDeployRequest,
        user_id: str
    ) -> Dict:
        """Deploy model (shadow/canary only for defensive system)"""
        
        model = self.models.get(model_id)
        if not model:
            raise ValueError("Model not found")
        
        if model.status != ModelStatus.READY:
            raise ValueError("Model not ready for deployment")
        
        # Only allow shadow or canary modes (no production execution)
        if request.mode == DeploymentMode.PRODUCTION:
            raise ValueError("Production deployment not allowed in defensive-only mode")
        
        deployment_id = str(uuid.uuid4())[:8]
        
        self.deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "mode": request.mode,
            "canary_percent": request.canary_percent,
            "deployed_at": datetime.now(),
            "deployed_by": user_id,
            "status": "active",
            "metrics": {
                "predictions": 0,
                "errors": 0,
                "avg_latency_ms": 0
            }
        }
        
        # Update model status
        model.status = ModelStatus.DEPLOYED
        
        return self.deployments[deployment_id]


# Global training service
training_service = TrainingService()


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_role(UserRole.ML_ENGINEER))
) -> TrainingResponse:
    """
    Start model training job
    Requires ML_ENGINEER role
    """
    
    # Validate dataset query
    pool = await get_clickhouse_pool()
    is_valid, error = await pool.validate_query(request.dataset_query)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid query: {error}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Start training in background
    background_tasks.add_task(
        training_service.start_training,
        job_id=job_id,
        request=request,
        user_id=current_user.user_id
    )
    
    # Return immediate response
    return TrainingResponse(
        success=True,
        job_id=job_id,
        model_id=f"model_{job_id[:8]}",
        status=ModelStatus.TRAINING,
        started_at=datetime.now(),
        progress_percent=0.0,
        message="Training job started"
    )


@router.get("/jobs/{job_id}", response_model=TrainingResponse)
async def get_training_status(
    job_id: str,
    current_user: TokenData = Depends(require_role(UserRole.ML_ENGINEER))
) -> TrainingResponse:
    """Get training job status"""
    
    job_info = training_service.get_job_status(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if job_info["user_id"] != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return TrainingResponse(
        success=True,
        job_id=job_id,
        model_id=job_info["model_id"],
        status=job_info["status"],
        started_at=job_info["started_at"],
        estimated_completion=job_info.get("estimated_completion"),
        progress_percent=job_info["progress"],
        metrics=job_info.get("metrics"),
        message=job_info.get("error") if job_info["status"] == ModelStatus.FAILED else "Training in progress"
    )


@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    limit: int = 20,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> List[ModelInfo]:
    """List trained models"""
    
    # ML Engineers and Admins can see all models
    if current_user.role in [UserRole.ML_ENGINEER, UserRole.ADMIN]:
        models = training_service.list_models()
    else:
        # Others can only see deployed models
        models = [
            m for m in training_service.list_models()
            if m.status == ModelStatus.DEPLOYED
        ]
    
    return models[:limit]


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> ModelInfo:
    """Get model information"""
    
    model = training_service.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check access
    if model.status != ModelStatus.DEPLOYED:
        if current_user.role not in [UserRole.ML_ENGINEER, UserRole.ADMIN]:
            raise HTTPException(status_code=403, detail="Access denied")
    
    return model


@router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    request: ModelDeployRequest,
    current_user: TokenData = Depends(require_role(UserRole.ML_ENGINEER))
) -> Dict[str, Any]:
    """
    Deploy model in shadow or canary mode
    Production deployment disabled for defensive-only system
    """
    
    try:
        deployment = await training_service.deploy_model(
            model_id=model_id,
            request=request,
            user_id=current_user.user_id
        )
        
        return {
            "success": True,
            "deployment": deployment,
            "message": f"Model deployed in {request.mode.value} mode"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: TokenData = Depends(require_role(UserRole.ML_ENGINEER))
) -> Dict[str, Any]:
    """Delete a model"""
    
    model = training_service.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check ownership
    if model.trained_by != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Don't delete deployed models
    if model.status == ModelStatus.DEPLOYED:
        raise HTTPException(status_code=400, detail="Cannot delete deployed model")
    
    # Delete model
    del training_service.models[model_id]
    
    return {
        "success": True,
        "message": f"Model {model_id} deleted"
    }


@router.get("/templates")
async def get_training_templates(
    current_user: TokenData = Depends(require_role(UserRole.ML_ENGINEER))
) -> Dict[str, Any]:
    """Get predefined training templates"""
    
    templates = {
        "arbitrage_detector": {
            "name": "Arbitrage Detector",
            "description": "Detect profitable arbitrage opportunities",
            "model_type": "xgboost",
            "dataset_query": """
                SELECT 
                    roi_pct,
                    legs,
                    toFloat32(arrayElement(splitByChar(',', dex_route), 1) != '') as uses_raydium,
                    toFloat32(arrayElement(splitByChar(',', dex_route), 2) != '') as uses_orca,
                    confidence,
                    slot % 1000 as slot_mod,
                    toHour(detected_at) as hour_of_day,
                    IF(roi_pct > 0.5, 1, 0) as is_profitable
                FROM mev.arbitrage_alerts
                WHERE detected_at >= now() - INTERVAL 7 DAY
            """,
            "features": [
                "legs", "uses_raydium", "uses_orca", 
                "confidence", "slot_mod", "hour_of_day"
            ],
            "target": "is_profitable",
            "hyperparameters": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8
            }
        },
        "sandwich_predictor": {
            "name": "Sandwich Attack Predictor",
            "description": "Predict sandwich attack vulnerability",
            "model_type": "lightgbm",
            "dataset_query": """
                SELECT 
                    victim_loss,
                    attacker_profit,
                    victim_loss / nullif(attacker_profit, 0) as loss_profit_ratio,
                    slot % 1000 as slot_mod,
                    toHour(detected_at) as hour_of_day,
                    IF(victim_loss > 100, 1, 0) as high_impact
                FROM mev.sandwich_alerts
                WHERE detected_at >= now() - INTERVAL 7 DAY
            """,
            "features": [
                "victim_loss", "attacker_profit", "loss_profit_ratio",
                "slot_mod", "hour_of_day"
            ],
            "target": "high_impact",
            "hyperparameters": {
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "feature_fraction": 0.9
            }
        }
    }
    
    return {
        "success": True,
        "templates": templates
    }