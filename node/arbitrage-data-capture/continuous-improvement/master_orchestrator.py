#!/usr/bin/env python3
"""
Master Orchestrator for Continuous Improvement System
Production-grade coordinator for all optimization components
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitoring.real_time_monitor import RealTimeMonitor
from feedback.auto_feedback_loop import AutomatedFeedbackLoop
from optimization.system_optimizer import SystemOptimizer
from auditing.performance_auditor import PerformanceAuditor
from models.advanced_features import AdvancedFeaturesOrchestrator

import yaml
import logging
from pathlib import Path
from datetime import datetime
import signal
import uvloop
from prometheus_client import start_http_server
import warnings
warnings.filterwarnings('ignore')

# Ultra-performance async
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MasterOrchestrator:
    """Master orchestrator for the entire continuous improvement system"""
    
    def __init__(self, config_path: str = "configs/system_config.yaml"):
        self.config = self._load_config(config_path)
        self.components = {}
        self.is_running = False
        self.tasks = []
        
    def _load_config(self, config_path: str) -> dict:
        """Load system configuration"""
        config_file = Path(__file__).parent / config_path
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    async def initialize_components(self):
        """Initialize all system components"""
        
        logger.info("Initializing Continuous Improvement System...")
        
        # Initialize monitoring
        if self.config['monitoring']['enabled']:
            self.components['monitor'] = RealTimeMonitor(
                redis_url=f"redis://{self.config['optimization']['database']['redis']['host']}:{self.config['optimization']['database']['redis']['port']}",
                kafka_bootstrap=self.config['optimization']['kafka']['bootstrap_servers'],
                clickhouse_host=self.config['optimization']['database']['clickhouse']['host']
            )
            await self.components['monitor'].initialize()
            logger.info("✓ Real-time monitoring initialized")
        
        # Initialize feedback loop
        if self.config['feedback_loop']['enabled']:
            self.components['feedback'] = AutomatedFeedbackLoop(
                clickhouse_host=self.config['optimization']['database']['clickhouse']['host'],
                redis_url=f"redis://{self.config['optimization']['database']['redis']['host']}:{self.config['optimization']['database']['redis']['port']}",
                kafka_bootstrap=self.config['optimization']['kafka']['bootstrap_servers'],
                mlflow_uri=self.config['feedback_loop']['model_management']['mlflow_uri']
            )
            await self.components['feedback'].initialize()
            logger.info("✓ Automated feedback loop initialized")
        
        # Initialize system optimizer
        if self.config['optimization']['enabled']:
            self.components['optimizer'] = SystemOptimizer()
            await self.components['optimizer'].initialize()
            logger.info("✓ System optimizer initialized")
        
        # Initialize performance auditor
        if self.config['auditing']['enabled']:
            self.components['auditor'] = PerformanceAuditor()
            logger.info("✓ Performance auditor initialized")
        
        # Initialize advanced features
        if self.config['advanced_features']['self_healing']['enabled']:
            self.components['advanced'] = AdvancedFeaturesOrchestrator()
            await self.components['advanced'].initialize()
            logger.info("✓ Advanced features initialized")
        
        # Start Prometheus metrics server
        if self.config['telemetry']['prometheus']['enabled']:
            start_http_server(self.config['telemetry']['prometheus']['port'])
            logger.info(f"✓ Prometheus metrics server started on port {self.config['telemetry']['prometheus']['port']}")
        
        logger.info("="*60)
        logger.info("CONTINUOUS IMPROVEMENT SYSTEM INITIALIZED")
        logger.info(f"Target Performance:")
        logger.info(f"  • Latency P99: <{self.config['performance_targets']['latency_p99_ms']}ms")
        logger.info(f"  • Throughput: {self.config['performance_targets']['throughput_rps']:,} RPS")
        logger.info(f"  • Availability: {self.config['performance_targets']['availability_percent']}%")
        logger.info("="*60)
    
    async def start_components(self):
        """Start all enabled components"""
        
        self.is_running = True
        
        # Start monitoring
        if 'monitor' in self.components:
            self.tasks.append(
                asyncio.create_task(self.components['monitor'].start())
            )
            logger.info("Started real-time monitoring")
        
        # Start feedback loop
        if 'feedback' in self.components:
            self.tasks.append(
                asyncio.create_task(self.components['feedback'].start())
            )
            logger.info("Started automated feedback loop")
        
        # Start system optimizer
        if 'optimizer' in self.components:
            self.tasks.append(
                asyncio.create_task(self.components['optimizer'].start())
            )
            logger.info("Started system optimizer")
        
        # Start performance auditor
        if 'auditor' in self.components:
            self.tasks.append(
                asyncio.create_task(self.components['auditor'].continuous_auditing())
            )
            logger.info("Started performance auditor")
        
        # Start advanced features
        if 'advanced' in self.components:
            self.tasks.append(
                asyncio.create_task(self.components['advanced'].autonomous_optimization_loop())
            )
            logger.info("Started advanced features")
        
        # Start health check loop
        self.tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
        
        logger.info("All components started successfully")
    
    async def _health_check_loop(self):
        """Periodic health check of all components"""
        
        while self.is_running:
            try:
                health_status = {
                    'timestamp': datetime.now().isoformat(),
                    'components': {}
                }
                
                # Check each component
                for name, component in self.components.items():
                    if hasattr(component, 'metrics'):
                        health_status['components'][name] = 'healthy'
                    else:
                        health_status['components'][name] = 'unknown'
                
                # Log health status
                healthy_count = sum(1 for status in health_status['components'].values() 
                                  if status == 'healthy')
                total_count = len(health_status['components'])
                
                logger.info(f"System Health: {healthy_count}/{total_count} components healthy")
                
                # Check performance targets
                if 'monitor' in self.components:
                    metrics = self.components['monitor'].metrics
                    
                    if metrics.avg_latency > self.config['performance_targets']['latency_p99_ms']:
                        logger.warning(f"Latency exceeds target: {metrics.avg_latency:.2f}ms")
                    
                    if metrics.throughput < self.config['performance_targets']['throughput_rps']:
                        logger.warning(f"Throughput below target: {metrics.throughput:.0f} RPS")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Gracefully stop all components"""
        
        logger.info("Stopping Continuous Improvement System...")
        self.is_running = False
        
        # Stop all components
        for name, component in self.components.items():
            if hasattr(component, 'stop'):
                await component.stop()
                logger.info(f"Stopped {name}")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("All components stopped successfully")
    
    async def run(self):
        """Main execution loop"""
        
        try:
            await self.initialize_components()
            await self.start_components()
            
            # Keep running until interrupted
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await self.stop()
            raise


async def main():
    """Main entry point"""
    
    orchestrator = MasterOrchestrator()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(orchestrator.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await orchestrator.stop()
    except Exception as e:
        logger.error(f"System error: {e}")
        await orchestrator.stop()
        sys.exit(1)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     ELITE CONTINUOUS IMPROVEMENT & FEEDBACK LOOP SYSTEM     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Performance Targets:                                        ║
    ║  • Latency: <5ms (P99)                                      ║
    ║  • Throughput: 100k+ TPS                                    ║
    ║  • Availability: 99.99%                                      ║
    ║  • Auto-retraining: <1 hour                                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Features:                                                   ║
    ║  ✓ Real-time model monitoring & drift detection             ║
    ║  ✓ Automated feedback loop with MLflow                      ║
    ║  ✓ System optimization with auto-scaling                    ║
    ║  ✓ Performance auditing & predictive analytics              ║
    ║  ✓ Self-healing & reinforcement learning                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())