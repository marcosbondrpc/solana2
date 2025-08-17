"""
Advanced Model Monitoring and MLOps Pipeline
Real-time drift detection, performance monitoring, and automated retraining
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import json
import asyncio
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import shap

# Drift detection
from alibi_detect.cd import KSDrift, MMDDrift, ChiSquareDrift, TabularDrift
from alibi_detect.cd import ClassifierDrift, SpotTheDiffDrift
from river import drift

# Database and streaming
import clickhouse_driver
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import redis
import aioredis

# Monitoring and alerting
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import slack_sdk
from slack_sdk.webhook import WebhookClient

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring"""
    
    # Drift detection
    drift_detection_method: str = 'ks'  # 'ks', 'mmd', 'chi2', 'tabular'
    drift_threshold: float = 0.05
    drift_window_size: int = 1000
    reference_window_size: int = 10000
    
    # Performance monitoring
    performance_window: int = 1000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'accuracy': 0.90,
        'precision': 0.85,
        'recall': 0.85,
        'f1': 0.85,
        'latency_ms': 15.0
    })
    
    # Retraining triggers
    auto_retrain: bool = True
    retrain_on_drift: bool = True
    retrain_on_performance_drop: bool = True
    retrain_schedule: str = 'daily'  # 'hourly', 'daily', 'weekly'
    min_samples_for_retrain: int = 10000
    
    # Alerting
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ['email', 'slack', 'prometheus'])
    email_config: Dict[str, str] = field(default_factory=dict)
    slack_webhook_url: str = None
    
    # Data storage
    clickhouse_host: str = 'localhost'
    clickhouse_port: int = 9000
    redis_host: str = 'localhost'
    redis_port: int = 6379
    kafka_bootstrap_servers: str = 'localhost:9092'
    
    # Monitoring intervals
    monitor_interval_seconds: int = 60
    metric_aggregation_window: int = 300  # 5 minutes


class DriftDetector:
    """Advanced drift detection for features and predictions"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_data = None
        self.drift_detectors = {}
        self.drift_history = []
        
    def initialize_detectors(self, reference_data: np.ndarray, feature_names: List[str]):
        """Initialize drift detectors with reference data"""
        self.reference_data = reference_data
        
        # KS-Test detector for each feature
        for i, feature_name in enumerate(feature_names):
            self.drift_detectors[f'ks_{feature_name}'] = KSDrift(
                reference_data[:, i:i+1],
                p_val=self.config.drift_threshold,
                alternative='two-sided'
            )
        
        # MMD detector for multivariate drift
        self.drift_detectors['mmd_multivariate'] = MMDDrift(
            reference_data,
            p_val=self.config.drift_threshold,
            n_permutations=100
        )
        
        # Tabular drift detector
        self.drift_detectors['tabular'] = TabularDrift(
            reference_data,
            p_val=self.config.drift_threshold,
            categories_per_feature={},
            n_permutations=100
        )
        
        logger.info(f"Initialized {len(self.drift_detectors)} drift detectors")
    
    def detect_drift(self, current_data: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Detect drift in current data"""
        drift_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': False,
            'feature_drifts': {},
            'multivariate_drift': False,
            'drift_scores': {}
        }
        
        # Check individual feature drift
        for i, feature_name in enumerate(feature_names):
            detector_name = f'ks_{feature_name}'
            if detector_name in self.drift_detectors:
                detector = self.drift_detectors[detector_name]
                result = detector.predict(current_data[:, i:i+1])
                
                drift_results['feature_drifts'][feature_name] = {
                    'drift': bool(result['data']['is_drift']),
                    'p_value': float(result['data']['p_val']),
                    'threshold': self.config.drift_threshold
                }
                
                if result['data']['is_drift']:
                    drift_results['drift_detected'] = True
        
        # Check multivariate drift
        if 'mmd_multivariate' in self.drift_detectors:
            mmd_result = self.drift_detectors['mmd_multivariate'].predict(current_data)
            drift_results['multivariate_drift'] = bool(mmd_result['data']['is_drift'])
            drift_results['drift_scores']['mmd'] = float(mmd_result['data']['p_val'])
            
            if mmd_result['data']['is_drift']:
                drift_results['drift_detected'] = True
        
        # Additional statistical tests
        if self.reference_data is not None:
            # Wasserstein distance for distribution shift
            for i, feature_name in enumerate(feature_names):
                ref_feature = self.reference_data[:, i]
                curr_feature = current_data[:, i]
                w_distance = wasserstein_distance(ref_feature, curr_feature)
                drift_results['drift_scores'][f'wasserstein_{feature_name}'] = w_distance
        
        # Store drift history
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift report"""
        if not self.drift_history:
            return {}
        
        report = {
            'total_checks': len(self.drift_history),
            'drift_rate': sum(1 for d in self.drift_history if d['drift_detected']) / len(self.drift_history),
            'feature_drift_rates': {},
            'recent_drifts': self.drift_history[-10:],
            'recommendations': []
        }
        
        # Calculate per-feature drift rates
        feature_drifts = {}
        for drift_result in self.drift_history:
            for feature, result in drift_result.get('feature_drifts', {}).items():
                if feature not in feature_drifts:
                    feature_drifts[feature] = []
                feature_drifts[feature].append(result['drift'])
        
        for feature, drifts in feature_drifts.items():
            report['feature_drift_rates'][feature] = sum(drifts) / len(drifts)
        
        # Generate recommendations
        if report['drift_rate'] > 0.1:
            report['recommendations'].append("High drift rate detected. Consider retraining the model.")
        
        high_drift_features = [f for f, rate in report['feature_drift_rates'].items() if rate > 0.15]
        if high_drift_features:
            report['recommendations'].append(f"Features with high drift: {', '.join(high_drift_features)}")
        
        return report


class PerformanceMonitor:
    """Real-time model performance monitoring"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.performance_history = []
        self.prediction_buffer = []
        self.ground_truth_buffer = []
        
        # Prometheus metrics
        self.accuracy_gauge = Gauge('model_accuracy_current', 'Current model accuracy')
        self.precision_gauge = Gauge('model_precision_current', 'Current model precision')
        self.recall_gauge = Gauge('model_recall_current', 'Current model recall')
        self.f1_gauge = Gauge('model_f1_current', 'Current model F1 score')
        self.latency_histogram = Histogram('model_latency_ms', 'Model inference latency')
        
    def add_prediction(self, prediction: int, probability: float, latency_ms: float, 
                      features: Dict[str, Any]):
        """Add a prediction to the buffer"""
        self.prediction_buffer.append({
            'timestamp': datetime.utcnow(),
            'prediction': prediction,
            'probability': probability,
            'latency_ms': latency_ms,
            'features': features
        })
        
        # Update latency metric
        self.latency_histogram.observe(latency_ms)
        
        # Trim buffer if too large
        if len(self.prediction_buffer) > self.config.performance_window * 2:
            self.prediction_buffer = self.prediction_buffer[-self.config.performance_window:]
    
    def add_ground_truth(self, prediction_id: str, true_label: int):
        """Add ground truth for a prediction"""
        self.ground_truth_buffer.append({
            'timestamp': datetime.utcnow(),
            'prediction_id': prediction_id,
            'true_label': true_label
        })
        
        # Trim buffer
        if len(self.ground_truth_buffer) > self.config.performance_window * 2:
            self.ground_truth_buffer = self.ground_truth_buffer[-self.config.performance_window:]
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics"""
        if len(self.prediction_buffer) < 100 or len(self.ground_truth_buffer) < 100:
            return {}
        
        # Match predictions with ground truth
        matched_pairs = []
        for gt in self.ground_truth_buffer[-self.config.performance_window:]:
            # Find corresponding prediction (simplified - in practice use prediction_id)
            for pred in self.prediction_buffer:
                if abs((pred['timestamp'] - gt['timestamp']).total_seconds()) < 1:
                    matched_pairs.append((pred['prediction'], gt['true_label']))
                    break
        
        if len(matched_pairs) < 10:
            return {}
        
        y_pred, y_true = zip(*matched_pairs)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'samples': len(matched_pairs)
        }
        
        # Calculate average latency
        recent_predictions = self.prediction_buffer[-self.config.performance_window:]
        if recent_predictions:
            metrics['avg_latency_ms'] = np.mean([p['latency_ms'] for p in recent_predictions])
        
        # Update Prometheus metrics
        self.accuracy_gauge.set(metrics['accuracy'])
        self.precision_gauge.set(metrics['precision'])
        self.recall_gauge.set(metrics['recall'])
        self.f1_gauge.set(metrics['f1'])
        
        # Store in history
        self.performance_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def check_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check if any alerts should be triggered"""
        alerts = []
        
        for metric_name, threshold in self.config.alert_thresholds.items():
            if metric_name in metrics:
                if metric_name == 'latency_ms':
                    # For latency, alert if above threshold
                    if metrics[metric_name] > threshold:
                        alerts.append({
                            'type': 'performance',
                            'severity': 'high' if metrics[metric_name] > threshold * 1.5 else 'medium',
                            'metric': metric_name,
                            'value': metrics[metric_name],
                            'threshold': threshold,
                            'message': f"Model latency ({metrics[metric_name]:.2f}ms) exceeds threshold ({threshold}ms)"
                        })
                else:
                    # For other metrics, alert if below threshold
                    if metrics[metric_name] < threshold:
                        alerts.append({
                            'type': 'performance',
                            'severity': 'high' if metrics[metric_name] < threshold * 0.8 else 'medium',
                            'metric': metric_name,
                            'value': metrics[metric_name],
                            'threshold': threshold,
                            'message': f"Model {metric_name} ({metrics[metric_name]:.3f}) below threshold ({threshold})"
                        })
        
        return alerts
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {}
        
        recent_metrics = [h['metrics'] for h in self.performance_history[-100:]]
        
        report = {
            'current_metrics': self.calculate_metrics(),
            'historical_metrics': {
                'accuracy': {
                    'mean': np.mean([m.get('accuracy', 0) for m in recent_metrics]),
                    'std': np.std([m.get('accuracy', 0) for m in recent_metrics]),
                    'trend': self._calculate_trend([m.get('accuracy', 0) for m in recent_metrics])
                },
                'precision': {
                    'mean': np.mean([m.get('precision', 0) for m in recent_metrics]),
                    'std': np.std([m.get('precision', 0) for m in recent_metrics]),
                    'trend': self._calculate_trend([m.get('precision', 0) for m in recent_metrics])
                },
                'recall': {
                    'mean': np.mean([m.get('recall', 0) for m in recent_metrics]),
                    'std': np.std([m.get('recall', 0) for m in recent_metrics]),
                    'trend': self._calculate_trend([m.get('recall', 0) for m in recent_metrics])
                },
                'latency_ms': {
                    'mean': np.mean([m.get('avg_latency_ms', 0) for m in recent_metrics]),
                    'p50': np.percentile([m.get('avg_latency_ms', 0) for m in recent_metrics], 50),
                    'p95': np.percentile([m.get('avg_latency_ms', 0) for m in recent_metrics], 95),
                    'p99': np.percentile([m.get('avg_latency_ms', 0) for m in recent_metrics], 99)
                }
            },
            'total_predictions': sum([m.get('samples', 0) for m in recent_metrics]),
            'monitoring_period': f"{len(self.performance_history)} checks"
        }
        
        return report
    
    @staticmethod
    def _calculate_trend(values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if slope > 0.001:
            return 'improving'
        elif slope < -0.001:
            return 'degrading'
        else:
            return 'stable'


class AutoRetrainer:
    """Automated model retraining pipeline"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.retrain_history = []
        self.last_retrain = datetime.utcnow()
        
    async def check_retrain_conditions(self, drift_detected: bool, 
                                      performance_metrics: Dict[str, float]) -> bool:
        """Check if model should be retrained"""
        should_retrain = False
        reasons = []
        
        # Check drift condition
        if self.config.retrain_on_drift and drift_detected:
            should_retrain = True
            reasons.append("Data drift detected")
        
        # Check performance condition
        if self.config.retrain_on_performance_drop:
            for metric, threshold in self.config.alert_thresholds.items():
                if metric in performance_metrics:
                    if metric == 'latency_ms':
                        continue  # Skip latency for retraining decisions
                    
                    if performance_metrics[metric] < threshold * 0.9:  # 10% below threshold
                        should_retrain = True
                        reasons.append(f"{metric} below threshold ({performance_metrics[metric]:.3f} < {threshold * 0.9:.3f})")
        
        # Check schedule
        if self.config.retrain_schedule:
            time_since_last = datetime.utcnow() - self.last_retrain
            
            if self.config.retrain_schedule == 'hourly' and time_since_last > timedelta(hours=1):
                should_retrain = True
                reasons.append("Scheduled hourly retrain")
            elif self.config.retrain_schedule == 'daily' and time_since_last > timedelta(days=1):
                should_retrain = True
                reasons.append("Scheduled daily retrain")
            elif self.config.retrain_schedule == 'weekly' and time_since_last > timedelta(days=7):
                should_retrain = True
                reasons.append("Scheduled weekly retrain")
        
        if should_retrain:
            logger.info(f"Retraining triggered: {', '.join(reasons)}")
            
        return should_retrain, reasons
    
    async def trigger_retrain(self, reasons: List[str]) -> Dict[str, Any]:
        """Trigger model retraining"""
        retrain_job = {
            'job_id': datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.utcnow().isoformat(),
            'reasons': reasons,
            'status': 'initiated'
        }
        
        try:
            # Here you would trigger your actual retraining pipeline
            # This could be a Kubernetes job, Airflow DAG, etc.
            
            # For now, we'll simulate the retraining process
            logger.info(f"Starting retrain job {retrain_job['job_id']}")
            
            # Record retrain
            self.retrain_history.append(retrain_job)
            self.last_retrain = datetime.utcnow()
            
            retrain_job['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            retrain_job['status'] = 'failed'
            retrain_job['error'] = str(e)
        
        return retrain_job


class AlertManager:
    """Manage and send alerts through various channels"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = []
        self.slack_client = None
        
        if self.config.slack_webhook_url:
            self.slack_client = WebhookClient(self.config.slack_webhook_url)
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        alert['timestamp'] = datetime.utcnow().isoformat()
        self.alert_history.append(alert)
        
        if not self.config.enable_alerts:
            return
        
        for channel in self.config.alert_channels:
            try:
                if channel == 'email':
                    await self._send_email_alert(alert)
                elif channel == 'slack':
                    await self._send_slack_alert(alert)
                elif channel == 'prometheus':
                    self._record_prometheus_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    async def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert"""
        if not self.config.email_config:
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.config.email_config.get('from')
        msg['To'] = self.config.email_config.get('to')
        msg['Subject'] = f"[{alert['severity'].upper()}] ML Model Alert: {alert['type']}"
        
        body = f"""
        Alert Type: {alert['type']}
        Severity: {alert['severity']}
        Message: {alert['message']}
        Timestamp: {alert['timestamp']}
        
        Details:
        {json.dumps(alert, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email (would need SMTP configuration)
        # server = smtplib.SMTP(self.config.email_config.get('smtp_server'), 587)
        # server.send_message(msg)
    
    async def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert"""
        if not self.slack_client:
            return
        
        color = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'good'
        }.get(alert['severity'], 'warning')
        
        response = self.slack_client.send(
            text=f"ML Model Alert: {alert['type']}",
            attachments=[{
                'color': color,
                'title': alert['message'],
                'fields': [
                    {'title': 'Severity', 'value': alert['severity'], 'short': True},
                    {'title': 'Type', 'value': alert['type'], 'short': True},
                    {'title': 'Timestamp', 'value': alert['timestamp'], 'short': False}
                ]
            }]
        )
    
    def _record_prometheus_alert(self, alert: Dict[str, Any]):
        """Record alert in Prometheus"""
        alert_counter = Counter(f"ml_model_alert_{alert['type']}", "Model alerts by type")
        alert_counter.inc()


class ModelMonitor:
    """Main monitoring orchestrator"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.drift_detector = DriftDetector(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.auto_retrainer = AutoRetrainer(self.config)
        self.alert_manager = AlertManager(self.config)
        self.running = False
        
    async def start_monitoring(self):
        """Start the monitoring loop"""
        self.running = True
        logger.info("Starting model monitoring...")
        
        while self.running:
            try:
                # Collect current metrics
                metrics = self.performance_monitor.calculate_metrics()
                
                # Check for alerts
                alerts = self.performance_monitor.check_alerts(metrics)
                for alert in alerts:
                    await self.alert_manager.send_alert(alert)
                
                # Check drift (would need actual data)
                # drift_results = self.drift_detector.detect_drift(current_data, feature_names)
                
                # Check retrain conditions
                if self.config.auto_retrain:
                    should_retrain, reasons = await self.auto_retrainer.check_retrain_conditions(
                        drift_detected=False,  # Would come from drift_results
                        performance_metrics=metrics
                    )
                    
                    if should_retrain:
                        await self.auto_retrainer.trigger_retrain(reasons)
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config.monitor_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard"""
        return {
            'performance': self.performance_monitor.get_performance_report(),
            'drift': self.drift_detector.get_drift_report(),
            'alerts': self.alert_manager.alert_history[-20:],
            'retrains': self.auto_retrainer.retrain_history[-10:],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.running = False
        logger.info("Stopping model monitoring...")