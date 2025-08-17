"""
Advanced Automated Report Generation System for Arbitrage Detection Infrastructure
Handles all reporting needs from real-time to historical analysis with sub-5 second generation
"""

import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from clickhouse_driver import Client as ClickHouseClient
import redis.asyncio as redis
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Summary
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import xlsxwriter
from openpyxl import Workbook
from openpyxl.styles import Font, Fill, Border, Side, Alignment, PatternFill
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.utils import get_column_letter
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
from typing_extensions import Protocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics for monitoring report generation
report_generation_time = Histogram(
    'report_generation_duration_seconds',
    'Time taken to generate reports',
    ['report_type', 'format']
)
reports_generated = Counter(
    'reports_generated_total',
    'Total number of reports generated',
    ['report_type', 'format']
)
report_errors = Counter(
    'report_generation_errors_total',
    'Total number of report generation errors',
    ['report_type', 'error_type']
)


class ReportType(Enum):
    """Types of reports available in the system"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    REAL_TIME = "realtime"
    CUSTOM = "custom"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    RISK = "risk"
    PERFORMANCE = "performance"
    ML_MODEL = "ml_model"
    COST_ANALYSIS = "cost_analysis"
    COMPETITION = "competition"
    OPPORTUNITY = "opportunity"


class ReportFormat(Enum):
    """Output formats for reports"""
    PDF = "pdf"
    EXCEL = "excel"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    POWERPOINT = "pptx"
    INTERACTIVE_HTML = "interactive_html"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_type: ReportType
    format: ReportFormat
    start_date: datetime
    end_date: datetime
    filters: Dict[str, Any] = field(default_factory=dict)
    include_charts: bool = True
    include_raw_data: bool = False
    recipients: List[str] = field(default_factory=list)
    schedule: Optional[str] = None  # cron expression
    template: Optional[str] = None
    custom_metrics: List[str] = field(default_factory=list)
    aggregation_level: str = "daily"  # daily, hourly, minute
    comparison_period: Optional[str] = None  # previous_period, year_over_year
    
    
@dataclass
class ReportMetrics:
    """Metrics collected for reports"""
    total_arbitrage_opportunities: int
    captured_opportunities: int
    total_profit: float
    total_loss: float
    net_profit: float
    roi: float
    capital_efficiency: float
    success_rate: float
    average_profit_per_trade: float
    gas_costs: float
    infrastructure_costs: float
    total_volume: float
    unique_pairs_traded: int
    best_performing_strategy: str
    worst_performing_strategy: str
    model_accuracy: float
    model_precision: float
    model_recall: float
    system_uptime: float
    average_latency: float
    peak_throughput: int
    error_rate: float
    alerts_triggered: int
    competition_metrics: Dict[str, Any]
    market_conditions: Dict[str, Any]
    risk_metrics: Dict[str, Any]


class DataCollector:
    """Collects data from various sources for report generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.clickhouse_client = ClickHouseClient(
            host=config['clickhouse']['host'],
            port=config['clickhouse']['port'],
            database=config['clickhouse']['database']
        )
        self.redis_client = None  # Will be initialized async
        self.prometheus_url = config['prometheus']['url']
        self.config = config
        
    async def initialize(self):
        """Initialize async connections"""
        self.redis_client = await redis.from_url(
            f"redis://{self.config['redis']['host']}:{self.config['redis']['port']}"
        )
        
    async def collect_arbitrage_data(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Collect arbitrage opportunity data from ClickHouse"""
        query = f"""
        SELECT 
            timestamp,
            dex_a,
            dex_b,
            token_pair,
            price_difference,
            potential_profit,
            actual_profit,
            gas_cost,
            execution_time_ms,
            success,
            failure_reason,
            strategy_type,
            capital_used,
            slippage,
            mev_competition_level,
            block_number,
            transaction_hash
        FROM arbitrage_opportunities
        WHERE timestamp >= '{start_date.isoformat()}'
          AND timestamp <= '{end_date.isoformat()}'
        """
        
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    query += f" AND {key} IN ({','.join(map(str, value))})"
                else:
                    query += f" AND {key} = '{value}'"
                    
        query += " ORDER BY timestamp DESC"
        
        result = self.clickhouse_client.execute(query)
        columns = [
            'timestamp', 'dex_a', 'dex_b', 'token_pair', 'price_difference',
            'potential_profit', 'actual_profit', 'gas_cost', 'execution_time_ms',
            'success', 'failure_reason', 'strategy_type', 'capital_used',
            'slippage', 'mev_competition_level', 'block_number', 'transaction_hash'
        ]
        
        return pd.DataFrame(result, columns=columns)
        
    async def collect_model_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect ML model performance metrics"""
        query = f"""
        SELECT 
            model_version,
            timestamp,
            accuracy,
            precision,
            recall,
            f1_score,
            auc_roc,
            inference_time_ms,
            predictions_count,
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
            feature_importance
        FROM ml_model_metrics
        WHERE timestamp >= '{start_date.isoformat()}'
          AND timestamp <= '{end_date.isoformat()}'
        ORDER BY timestamp DESC
        """
        
        result = self.clickhouse_client.execute(query)
        
        if not result:
            return {}
            
        df = pd.DataFrame(result, columns=[
            'model_version', 'timestamp', 'accuracy', 'precision', 'recall',
            'f1_score', 'auc_roc', 'inference_time_ms', 'predictions_count',
            'true_positives', 'false_positives', 'true_negatives',
            'false_negatives', 'feature_importance'
        ])
        
        return {
            'average_accuracy': df['accuracy'].mean(),
            'average_precision': df['precision'].mean(),
            'average_recall': df['recall'].mean(),
            'average_f1': df['f1_score'].mean(),
            'average_inference_time': df['inference_time_ms'].mean(),
            'total_predictions': df['predictions_count'].sum(),
            'model_versions': df['model_version'].unique().tolist(),
            'performance_trend': df.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
                'accuracy': 'mean',
                'precision': 'mean',
                'recall': 'mean'
            }).to_dict()
        }
        
    async def collect_system_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect system performance metrics from Prometheus"""
        metrics = {}
        
        # Query Prometheus for system metrics
        queries = {
            'avg_latency': 'avg(rate(http_request_duration_seconds_sum[5m])) / avg(rate(http_request_duration_seconds_count[5m]))',
            'error_rate': 'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
            'throughput': 'sum(rate(arbitrage_opportunities_processed_total[5m]))',
            'cpu_usage': 'avg(rate(process_cpu_seconds_total[5m])) * 100',
            'memory_usage': 'avg(process_resident_memory_bytes)',
            'uptime': '(time() - process_start_time_seconds) / 86400'
        }
        
        async with aiohttp.ClientSession() as session:
            for metric_name, query in queries.items():
                url = f"{self.prometheus_url}/api/v1/query_range"
                params = {
                    'query': query,
                    'start': int(start_date.timestamp()),
                    'end': int(end_date.timestamp()),
                    'step': '1h'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == 'success' and data['data']['result']:
                            values = data['data']['result'][0]['values']
                            metrics[metric_name] = {
                                'average': np.mean([float(v[1]) for v in values]),
                                'max': np.max([float(v[1]) for v in values]),
                                'min': np.min([float(v[1]) for v in values]),
                                'values': values
                            }
                            
        return metrics
        
    async def collect_competition_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze competition in MEV space"""
        query = f"""
        SELECT 
            competitor_address,
            COUNT(*) as opportunities_taken,
            SUM(profit) as total_profit,
            AVG(gas_price) as avg_gas_price,
            AVG(execution_time_ms) as avg_execution_time,
            COUNT(DISTINCT strategy_type) as unique_strategies
        FROM competitor_transactions
        WHERE timestamp >= '{start_date.isoformat()}'
          AND timestamp <= '{end_date.isoformat()}'
        GROUP BY competitor_address
        ORDER BY total_profit DESC
        LIMIT 20
        """
        
        result = self.clickhouse_client.execute(query)
        
        competitors = []
        for row in result:
            competitors.append({
                'address': row[0],
                'opportunities_taken': row[1],
                'total_profit': row[2],
                'avg_gas_price': row[3],
                'avg_execution_time': row[4],
                'unique_strategies': row[5]
            })
            
        return {
            'top_competitors': competitors,
            'total_competition_profit': sum(c['total_profit'] for c in competitors),
            'market_share': self._calculate_market_share(competitors),
            'competitive_intensity': len(competitors) / max(1, (end_date - start_date).days)
        }
        
    def _calculate_market_share(self, competitors: List[Dict]) -> Dict[str, float]:
        """Calculate market share distribution"""
        total_profit = sum(c['total_profit'] for c in competitors)
        if total_profit == 0:
            return {}
            
        return {
            c['address'][:10]: (c['total_profit'] / total_profit) * 100
            for c in competitors[:5]  # Top 5 competitors
        }


class ReportGenerator:
    """Main report generation engine with advanced visualization and formatting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_collector = DataCollector(config)
        self.template_env = Environment(
            loader=FileSystemLoader('/home/kidgordones/0solana/node/arbitrage-data-capture/documentation-reporting/templates')
        )
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self):
        """Initialize async components"""
        await self.data_collector.initialize()
        
    async def generate_report(
        self,
        report_config: ReportConfig
    ) -> Tuple[str, bytes]:
        """Generate a report based on configuration"""
        start_time = time.time()
        
        try:
            # Collect all necessary data
            data = await self._collect_report_data(report_config)
            
            # Calculate metrics
            metrics = self._calculate_metrics(data)
            
            # Generate visualizations
            charts = await self._generate_visualizations(data, metrics)
            
            # Generate report in requested format
            report_content = await self._format_report(
                data, metrics, charts, report_config
            )
            
            # Record metrics
            generation_time = time.time() - start_time
            report_generation_time.labels(
                report_type=report_config.report_type.value,
                format=report_config.format.value
            ).observe(generation_time)
            
            reports_generated.labels(
                report_type=report_config.report_type.value,
                format=report_config.format.value
            ).inc()
            
            logger.info(
                f"Generated {report_config.report_type.value} report in "
                f"{report_config.format.value} format in {generation_time:.2f}s"
            )
            
            return f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}", report_content
            
        except Exception as e:
            report_errors.labels(
                report_type=report_config.report_type.value,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Error generating report: {e}")
            raise
            
    async def _collect_report_data(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Collect all data needed for the report"""
        tasks = []
        
        # Collect arbitrage data
        tasks.append(
            self.data_collector.collect_arbitrage_data(
                config.start_date,
                config.end_date,
                config.filters
            )
        )
        
        # Collect model metrics
        tasks.append(
            self.data_collector.collect_model_metrics(
                config.start_date,
                config.end_date
            )
        )
        
        # Collect system metrics
        tasks.append(
            self.data_collector.collect_system_metrics(
                config.start_date,
                config.end_date
            )
        )
        
        # Collect competition metrics
        tasks.append(
            self.data_collector.collect_competition_metrics(
                config.start_date,
                config.end_date
            )
        )
        
        results = await asyncio.gather(*tasks)
        
        return {
            'arbitrage_data': results[0],
            'model_metrics': results[1],
            'system_metrics': results[2],
            'competition_metrics': results[3]
        }
        
    def _calculate_metrics(self, data: Dict[str, Any]) -> ReportMetrics:
        """Calculate comprehensive metrics from collected data"""
        df = data['arbitrage_data']
        
        if df.empty:
            return ReportMetrics(
                total_arbitrage_opportunities=0,
                captured_opportunities=0,
                total_profit=0,
                total_loss=0,
                net_profit=0,
                roi=0,
                capital_efficiency=0,
                success_rate=0,
                average_profit_per_trade=0,
                gas_costs=0,
                infrastructure_costs=0,
                total_volume=0,
                unique_pairs_traded=0,
                best_performing_strategy="N/A",
                worst_performing_strategy="N/A",
                model_accuracy=0,
                model_precision=0,
                model_recall=0,
                system_uptime=0,
                average_latency=0,
                peak_throughput=0,
                error_rate=0,
                alerts_triggered=0,
                competition_metrics={},
                market_conditions={},
                risk_metrics={}
            )
            
        # Calculate basic metrics
        successful_trades = df[df['success'] == True]
        failed_trades = df[df['success'] == False]
        
        total_profit = successful_trades['actual_profit'].sum()
        total_loss = failed_trades['gas_cost'].sum()
        gas_costs = df['gas_cost'].sum()
        
        # Calculate advanced metrics
        capital_used = df['capital_used'].sum()
        roi = (total_profit - total_loss) / capital_used * 100 if capital_used > 0 else 0
        
        # Strategy performance
        strategy_performance = df.groupby('strategy_type')['actual_profit'].sum()
        best_strategy = strategy_performance.idxmax() if not strategy_performance.empty else "N/A"
        worst_strategy = strategy_performance.idxmin() if not strategy_performance.empty else "N/A"
        
        # Model metrics
        model_metrics = data.get('model_metrics', {})
        
        # System metrics
        system_metrics = data.get('system_metrics', {})
        
        return ReportMetrics(
            total_arbitrage_opportunities=len(df),
            captured_opportunities=len(successful_trades),
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=total_profit - total_loss - gas_costs,
            roi=roi,
            capital_efficiency=(total_profit / capital_used * 100) if capital_used > 0 else 0,
            success_rate=(len(successful_trades) / len(df) * 100) if len(df) > 0 else 0,
            average_profit_per_trade=successful_trades['actual_profit'].mean() if not successful_trades.empty else 0,
            gas_costs=gas_costs,
            infrastructure_costs=self._calculate_infrastructure_costs(data),
            total_volume=df['capital_used'].sum(),
            unique_pairs_traded=df['token_pair'].nunique(),
            best_performing_strategy=best_strategy,
            worst_performing_strategy=worst_strategy,
            model_accuracy=model_metrics.get('average_accuracy', 0),
            model_precision=model_metrics.get('average_precision', 0),
            model_recall=model_metrics.get('average_recall', 0),
            system_uptime=system_metrics.get('uptime', {}).get('average', 0),
            average_latency=system_metrics.get('avg_latency', {}).get('average', 0),
            peak_throughput=system_metrics.get('throughput', {}).get('max', 0),
            error_rate=system_metrics.get('error_rate', {}).get('average', 0),
            alerts_triggered=self._count_alerts(data),
            competition_metrics=data.get('competition_metrics', {}),
            market_conditions=self._analyze_market_conditions(df),
            risk_metrics=self._calculate_risk_metrics(df)
        )
        
    def _calculate_infrastructure_costs(self, data: Dict[str, Any]) -> float:
        """Calculate infrastructure costs based on usage"""
        # Simplified calculation - in production, integrate with cloud provider APIs
        base_cost = 100  # Base daily cost
        compute_cost = data.get('system_metrics', {}).get('cpu_usage', {}).get('average', 0) * 10
        storage_cost = len(data.get('arbitrage_data', [])) * 0.001
        
        return base_cost + compute_cost + storage_cost
        
    def _count_alerts(self, data: Dict[str, Any]) -> int:
        """Count number of alerts triggered"""
        # Simplified - in production, query alert system
        return 0
        
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions during reporting period"""
        if df.empty:
            return {}
            
        return {
            'volatility': df['price_difference'].std(),
            'average_spread': df['price_difference'].mean(),
            'competition_level': df['mev_competition_level'].mean() if 'mev_competition_level' in df else 0,
            'most_active_pairs': df['token_pair'].value_counts().head(5).to_dict(),
            'most_active_dexes': df.groupby('dex_a').size().sort_values(ascending=False).head(5).to_dict()
        }
        
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics"""
        if df.empty:
            return {}
            
        profits = df[df['actual_profit'] > 0]['actual_profit']
        losses = df[df['actual_profit'] < 0]['actual_profit'].abs()
        
        return {
            'max_drawdown': self._calculate_max_drawdown(df),
            'sharpe_ratio': self._calculate_sharpe_ratio(profits),
            'value_at_risk': np.percentile(losses, 95) if not losses.empty else 0,
            'expected_shortfall': losses[losses > np.percentile(losses, 95)].mean() if not losses.empty else 0,
            'win_loss_ratio': len(profits) / max(1, len(losses)) if not profits.empty else 0
        }
        
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if df.empty:
            return 0
            
        cumulative = df['actual_profit'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min()) * 100
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty:
            return 0
            
        avg_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0
            
        risk_free_rate = 0.02 / 365  # Daily risk-free rate
        
        return (avg_return - risk_free_rate) / std_return * np.sqrt(365)
        
    async def _generate_visualizations(
        self,
        data: Dict[str, Any],
        metrics: ReportMetrics
    ) -> Dict[str, Any]:
        """Generate comprehensive visualizations for the report"""
        charts = {}
        
        df = data['arbitrage_data']
        
        if not df.empty:
            # Profit/Loss over time
            charts['profit_timeline'] = self._create_profit_timeline(df)
            
            # Strategy performance comparison
            charts['strategy_performance'] = self._create_strategy_performance_chart(df)
            
            # DEX volume distribution
            charts['dex_distribution'] = self._create_dex_distribution_chart(df)
            
            # Success rate heatmap
            charts['success_heatmap'] = self._create_success_heatmap(df)
            
            # Gas cost analysis
            charts['gas_analysis'] = self._create_gas_analysis_chart(df)
            
            # Competition analysis
            charts['competition'] = self._create_competition_chart(data['competition_metrics'])
            
            # Model performance
            charts['model_performance'] = self._create_model_performance_chart(data['model_metrics'])
            
            # System metrics dashboard
            charts['system_dashboard'] = self._create_system_dashboard(data['system_metrics'])
            
        return charts
        
    def _create_profit_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create profit/loss timeline chart"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        daily_profit = df.groupby(df['timestamp'].dt.date).agg({
            'actual_profit': 'sum',
            'gas_cost': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Profit/Loss', 'Cumulative Profit'),
            vertical_spacing=0.1
        )
        
        # Daily profit/loss
        fig.add_trace(
            go.Bar(
                x=daily_profit['timestamp'],
                y=daily_profit['actual_profit'] - daily_profit['gas_cost'],
                name='Net Profit',
                marker_color=['green' if x > 0 else 'red' 
                             for x in daily_profit['actual_profit'] - daily_profit['gas_cost']]
            ),
            row=1, col=1
        )
        
        # Cumulative profit
        cumulative = (daily_profit['actual_profit'] - daily_profit['gas_cost']).cumsum()
        fig.add_trace(
            go.Scatter(
                x=daily_profit['timestamp'],
                y=cumulative,
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Profit Analysis',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_strategy_performance_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create strategy performance comparison chart"""
        strategy_stats = df.groupby('strategy_type').agg({
            'actual_profit': ['sum', 'mean', 'count'],
            'success': 'mean'
        }).round(2)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Total Profit by Strategy',
                'Success Rate by Strategy',
                'Average Profit per Trade',
                'Number of Trades'
            )
        )
        
        strategies = strategy_stats.index.tolist()
        
        # Total profit
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=strategy_stats[('actual_profit', 'sum')],
                name='Total Profit',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Success rate
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=strategy_stats[('success', 'mean')] * 100,
                name='Success Rate (%)',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # Average profit
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=strategy_stats[('actual_profit', 'mean')],
                name='Avg Profit',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Trade count
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=strategy_stats[('actual_profit', 'count')],
                name='Trade Count',
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Strategy Performance Analysis',
            height=700,
            showlegend=False
        )
        
        return fig
        
    def _create_dex_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create DEX volume distribution chart"""
        dex_volume = df.groupby('dex_a')['capital_used'].sum().sort_values(ascending=False).head(10)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=dex_volume.index,
                values=dex_volume.values,
                hole=0.3,
                textposition='inside',
                textinfo='percent+label'
            )
        ])
        
        fig.update_layout(
            title='Volume Distribution by DEX',
            height=500
        )
        
        return fig
        
    def _create_success_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create success rate heatmap by hour and day"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        
        success_matrix = df.pivot_table(
            values='success',
            index='hour',
            columns='day',
            aggfunc='mean'
        ) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=success_matrix.values,
            x=success_matrix.columns,
            y=success_matrix.index,
            colorscale='RdYlGn',
            text=success_matrix.values.round(1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Success Rate (%)")
        ))
        
        fig.update_layout(
            title='Success Rate Heatmap (Hour vs Day)',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            height=500
        )
        
        return fig
        
    def _create_gas_analysis_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create gas cost analysis chart"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        hourly_gas = df.groupby(df['timestamp'].dt.floor('H')).agg({
            'gas_cost': ['mean', 'max', 'min']
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hourly_gas['timestamp'],
            y=hourly_gas[('gas_cost', 'mean')],
            mode='lines',
            name='Average',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_gas['timestamp'],
            y=hourly_gas[('gas_cost', 'max')],
            mode='lines',
            name='Max',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_gas['timestamp'],
            y=hourly_gas[('gas_cost', 'min')],
            mode='lines',
            name='Min',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Gas Cost Analysis Over Time',
            xaxis_title='Time',
            yaxis_title='Gas Cost (ETH)',
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_competition_chart(self, competition_metrics: Dict[str, Any]) -> go.Figure:
        """Create competition analysis chart"""
        if not competition_metrics or not competition_metrics.get('top_competitors'):
            return go.Figure()
            
        competitors = competition_metrics['top_competitors'][:10]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top Competitors by Profit', 'Market Share Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Top competitors
        fig.add_trace(
            go.Bar(
                x=[c['address'][:10] for c in competitors],
                y=[c['total_profit'] for c in competitors],
                name='Profit',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Market share
        market_share = competition_metrics.get('market_share', {})
        if market_share:
            fig.add_trace(
                go.Pie(
                    labels=list(market_share.keys()),
                    values=list(market_share.values()),
                    hole=0.3
                ),
                row=1, col=2
            )
            
        fig.update_layout(
            title='Competition Analysis',
            height=400,
            showlegend=False
        )
        
        return fig
        
    def _create_model_performance_chart(self, model_metrics: Dict[str, Any]) -> go.Figure:
        """Create ML model performance chart"""
        if not model_metrics or 'performance_trend' not in model_metrics:
            return go.Figure()
            
        trend = model_metrics['performance_trend']
        
        fig = go.Figure()
        
        if 'accuracy' in trend:
            dates = list(trend['accuracy'].keys())
            fig.add_trace(go.Scatter(
                x=dates,
                y=list(trend['accuracy'].values()),
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue')
            ))
            
        if 'precision' in trend:
            fig.add_trace(go.Scatter(
                x=dates,
                y=list(trend['precision'].values()),
                mode='lines+markers',
                name='Precision',
                line=dict(color='green')
            ))
            
        if 'recall' in trend:
            fig.add_trace(go.Scatter(
                x=dates,
                y=list(trend['recall'].values()),
                mode='lines+markers',
                name='Recall',
                line=dict(color='red')
            ))
            
        fig.update_layout(
            title='ML Model Performance Trend',
            xaxis_title='Date',
            yaxis_title='Score',
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_system_dashboard(self, system_metrics: Dict[str, Any]) -> go.Figure:
        """Create system performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'CPU Usage (%)',
                'Memory Usage (GB)',
                'Throughput (ops/sec)',
                'Error Rate (%)'
            ),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # CPU Usage
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=system_metrics.get('cpu_usage', {}).get('average', 0),
                title={'text': "CPU %"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=1, col=1
        )
        
        # Memory Usage
        memory_gb = system_metrics.get('memory_usage', {}).get('average', 0) / (1024**3)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=memory_gb,
                title={'text': "Memory GB"},
                gauge={'axis': {'range': [None, 32]},
                       'bar': {'color': "green"}}
            ),
            row=1, col=2
        )
        
        # Throughput
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=system_metrics.get('throughput', {}).get('average', 0) * 1000,
                title={'text': "Throughput"},
                delta={'reference': 1000, 'relative': True}
            ),
            row=2, col=1
        )
        
        # Error Rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=system_metrics.get('error_rate', {}).get('average', 0) * 100,
                title={'text': "Error %"},
                gauge={'axis': {'range': [None, 5]},
                       'bar': {'color': "red"},
                       'steps': [
                           {'range': [0, 1], 'color': "lightgreen"},
                           {'range': [1, 3], 'color': "yellow"},
                           {'range': [3, 5], 'color': "red"}]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='System Performance Dashboard',
            height=600
        )
        
        return fig
        
    async def _format_report(
        self,
        data: Dict[str, Any],
        metrics: ReportMetrics,
        charts: Dict[str, Any],
        config: ReportConfig
    ) -> bytes:
        """Format report in requested output format"""
        if config.format == ReportFormat.PDF:
            return await self._generate_pdf_report(data, metrics, charts, config)
        elif config.format == ReportFormat.EXCEL:
            return await self._generate_excel_report(data, metrics, charts, config)
        elif config.format == ReportFormat.HTML:
            return await self._generate_html_report(data, metrics, charts, config)
        elif config.format == ReportFormat.INTERACTIVE_HTML:
            return await self._generate_interactive_html_report(data, metrics, charts, config)
        elif config.format == ReportFormat.JSON:
            return await self._generate_json_report(data, metrics, config)
        else:
            raise ValueError(f"Unsupported report format: {config.format}")
            
    async def _generate_pdf_report(
        self,
        data: Dict[str, Any],
        metrics: ReportMetrics,
        charts: Dict[str, Any],
        config: ReportConfig
    ) -> bytes:
        """Generate PDF report with charts and tables"""
        from io import BytesIO
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        story.append(Paragraph(
            f"{config.report_type.value.upper()} ARBITRAGE REPORT",
            title_style
        ))
        
        story.append(Spacer(1, 20))
        
        # Report period
        story.append(Paragraph(
            f"Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}",
            styles['Normal']
        ))
        
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Opportunities', f"{metrics.total_arbitrage_opportunities:,}"],
            ['Captured Opportunities', f"{metrics.captured_opportunities:,}"],
            ['Success Rate', f"{metrics.success_rate:.2f}%"],
            ['Net Profit', f"${metrics.net_profit:,.2f}"],
            ['ROI', f"{metrics.roi:.2f}%"],
            ['Capital Efficiency', f"{metrics.capital_efficiency:.2f}%"],
            ['Average Profit/Trade', f"${metrics.average_profit_per_trade:,.2f}"],
            ['Total Volume', f"${metrics.total_volume:,.2f}"],
            ['Unique Pairs Traded', f"{metrics.unique_pairs_traded:,}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(PageBreak())
        
        # Performance Analysis
        story.append(Paragraph("Performance Analysis", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Add charts (convert to images)
        for chart_name, chart in charts.items():
            if chart and hasattr(chart, 'write_image'):
                img_buffer = BytesIO()
                chart.write_image(img_buffer, format='png', width=700, height=400)
                img_buffer.seek(0)
                img = Image(img_buffer, width=6*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue()
        
    async def _generate_excel_report(
        self,
        data: Dict[str, Any],
        metrics: ReportMetrics,
        charts: Dict[str, Any],
        config: ReportConfig
    ) -> bytes:
        """Generate Excel report with multiple sheets"""
        from io import BytesIO
        
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#1a1a2e',
                'font_color': 'white',
                'border': 1
            })
            
            money_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # Summary sheet
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Opportunities',
                    'Captured Opportunities',
                    'Success Rate',
                    'Net Profit',
                    'ROI',
                    'Capital Efficiency',
                    'Average Profit/Trade',
                    'Total Volume'
                ],
                'Value': [
                    metrics.total_arbitrage_opportunities,
                    metrics.captured_opportunities,
                    metrics.success_rate,
                    metrics.net_profit,
                    metrics.roi,
                    metrics.capital_efficiency,
                    metrics.average_profit_per_trade,
                    metrics.total_volume
                ]
            })
            
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Arbitrage data sheet
            if not data['arbitrage_data'].empty:
                data['arbitrage_data'].to_excel(writer, sheet_name='Arbitrage Data', index=False)
                
            # Strategy performance sheet
            if not data['arbitrage_data'].empty:
                strategy_perf = data['arbitrage_data'].groupby('strategy_type').agg({
                    'actual_profit': ['sum', 'mean', 'count'],
                    'success': 'mean',
                    'gas_cost': 'mean'
                }).round(2)
                strategy_perf.to_excel(writer, sheet_name='Strategy Performance')
                
            # Competition sheet
            if data.get('competition_metrics', {}).get('top_competitors'):
                comp_df = pd.DataFrame(data['competition_metrics']['top_competitors'])
                comp_df.to_excel(writer, sheet_name='Competition', index=False)
                
            # Format sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:Z', 15)
                
        buffer.seek(0)
        return buffer.getvalue()
        
    async def _generate_html_report(
        self,
        data: Dict[str, Any],
        metrics: ReportMetrics,
        charts: Dict[str, Any],
        config: ReportConfig
    ) -> bytes:
        """Generate HTML report using templates"""
        template = self.template_env.get_template('report_template.html')
        
        # Convert charts to HTML
        charts_html = {}
        for name, chart in charts.items():
            if chart and hasattr(chart, 'to_html'):
                charts_html[name] = chart.to_html(include_plotlyjs='cdn')
                
        html_content = template.render(
            report_type=config.report_type.value,
            start_date=config.start_date,
            end_date=config.end_date,
            metrics=metrics,
            charts=charts_html,
            data=data
        )
        
        return html_content.encode('utf-8')
        
    async def _generate_interactive_html_report(
        self,
        data: Dict[str, Any],
        metrics: ReportMetrics,
        charts: Dict[str, Any],
        config: ReportConfig
    ) -> bytes:
        """Generate interactive HTML dashboard"""
        import dash
        from dash import dcc, html, dash_table
        import dash_bootstrap_components as dbc
        
        # Create Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(f"{config.report_type.value.upper()} Arbitrage Report"),
                    html.Hr()
                ])
            ]),
            
            # Metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Net Profit"),
                            html.H2(f"${metrics.net_profit:,.2f}")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("ROI"),
                            html.H2(f"{metrics.roi:.2f}%")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Success Rate"),
                            html.H2(f"{metrics.success_rate:.2f}%")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Volume"),
                            html.H2(f"${metrics.total_volume:,.2f}")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=charts.get('profit_timeline', {}))
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=charts.get('strategy_performance', {}))
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=charts.get('dex_distribution', {}))
                ], width=6)
            ], className="mb-4"),
            
            # Data table
            dbc.Row([
                dbc.Col([
                    html.H3("Recent Arbitrage Opportunities"),
                    dash_table.DataTable(
                        data=data['arbitrage_data'].head(100).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in data['arbitrage_data'].columns],
                        page_size=20,
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'success', 'filter_query': '{success} = True'},
                                'backgroundColor': 'green',
                                'color': 'white',
                            }
                        ]
                    )
                ])
            ])
        ], fluid=True)
        
        # Export to static HTML
        from dash import Dash
        import plotly.io as pio
        
        html_string = app.index_string
        for name, chart in charts.items():
            if chart:
                html_string = html_string.replace(
                    f'id="{name}"',
                    f'id="{name}">{chart.to_html(include_plotlyjs=False)}'
                )
                
        return html_string.encode('utf-8')
        
    async def _generate_json_report(
        self,
        data: Dict[str, Any],
        metrics: ReportMetrics,
        config: ReportConfig
    ) -> bytes:
        """Generate JSON report for API consumption"""
        report_dict = {
            'metadata': {
                'report_type': config.report_type.value,
                'start_date': config.start_date.isoformat(),
                'end_date': config.end_date.isoformat(),
                'generated_at': datetime.now().isoformat(),
                'filters': config.filters
            },
            'metrics': {
                'total_opportunities': metrics.total_arbitrage_opportunities,
                'captured_opportunities': metrics.captured_opportunities,
                'success_rate': metrics.success_rate,
                'net_profit': metrics.net_profit,
                'roi': metrics.roi,
                'capital_efficiency': metrics.capital_efficiency,
                'average_profit_per_trade': metrics.average_profit_per_trade,
                'total_volume': metrics.total_volume,
                'unique_pairs_traded': metrics.unique_pairs_traded,
                'gas_costs': metrics.gas_costs,
                'infrastructure_costs': metrics.infrastructure_costs,
                'best_strategy': metrics.best_performing_strategy,
                'worst_strategy': metrics.worst_performing_strategy,
                'model_accuracy': metrics.model_accuracy,
                'system_uptime': metrics.system_uptime,
                'average_latency': metrics.average_latency
            },
            'competition_metrics': metrics.competition_metrics,
            'market_conditions': metrics.market_conditions,
            'risk_metrics': metrics.risk_metrics
        }
        
        if config.include_raw_data:
            report_dict['raw_data'] = {
                'arbitrage_data': data['arbitrage_data'].to_dict('records'),
                'model_metrics': data['model_metrics'],
                'system_metrics': data['system_metrics']
            }
            
        return json.dumps(report_dict, default=str, indent=2).encode('utf-8')


class ReportScheduler:
    """Handles automated report scheduling and distribution"""
    
    def __init__(self, generator: ReportGenerator, config: Dict[str, Any]):
        self.generator = generator
        self.config = config
        self.scheduled_reports = []
        
    def schedule_report(self, report_config: ReportConfig):
        """Schedule a report for regular generation"""
        if report_config.schedule:
            schedule.every().day.at(report_config.schedule).do(
                lambda: asyncio.run(self._generate_and_send_report(report_config))
            )
            self.scheduled_reports.append(report_config)
            logger.info(f"Scheduled {report_config.report_type.value} report at {report_config.schedule}")
            
    async def _generate_and_send_report(self, report_config: ReportConfig):
        """Generate and send scheduled report"""
        try:
            # Update dates for current period
            if report_config.report_type == ReportType.DAILY:
                report_config.start_date = datetime.now() - timedelta(days=1)
                report_config.end_date = datetime.now()
            elif report_config.report_type == ReportType.WEEKLY:
                report_config.start_date = datetime.now() - timedelta(weeks=1)
                report_config.end_date = datetime.now()
            elif report_config.report_type == ReportType.MONTHLY:
                report_config.start_date = datetime.now() - timedelta(days=30)
                report_config.end_date = datetime.now()
                
            # Generate report
            filename, content = await self.generator.generate_report(report_config)
            
            # Send to recipients
            if report_config.recipients:
                await self._send_report_email(
                    filename,
                    content,
                    report_config
                )
                
            logger.info(f"Successfully generated and sent {report_config.report_type.value} report")
            
        except Exception as e:
            logger.error(f"Error generating scheduled report: {e}")
            
    async def _send_report_email(
        self,
        filename: str,
        content: bytes,
        config: ReportConfig
    ):
        """Send report via email"""
        msg = MIMEMultipart()
        msg['From'] = self.config['email']['from_address']
        msg['To'] = ', '.join(config.recipients)
        msg['Subject'] = f"{config.report_type.value.upper()} Arbitrage Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
        Please find attached the {config.report_type.value} arbitrage report for the period:
        {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}
        
        This is an automated report generated by the Arbitrage Detection Infrastructure.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach report file
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(content)
        encoders.encode_base64(attachment)
        
        ext = config.format.value
        attachment.add_header(
            'Content-Disposition',
            f'attachment; filename={filename}.{ext}'
        )
        
        msg.attach(attachment)
        
        # Send email
        with smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port']) as server:
            server.starttls()
            server.login(
                self.config['email']['username'],
                self.config['email']['password']
            )
            server.send_message(msg)
            
    def run_scheduler(self):
        """Run the report scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == "__main__":
    # Load configuration
    with open('/home/kidgordones/0solana/node/arbitrage-data-capture/documentation-reporting/configs/report_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize report generator
    generator = ReportGenerator(config)
    asyncio.run(generator.initialize())
    
    # Create scheduler
    scheduler = ReportScheduler(generator, config)
    
    # Schedule regular reports
    daily_config = ReportConfig(
        report_type=ReportType.DAILY,
        format=ReportFormat.PDF,
        start_date=datetime.now() - timedelta(days=1),
        end_date=datetime.now(),
        recipients=config['report_recipients']['daily'],
        schedule="09:00"
    )
    scheduler.schedule_report(daily_config)
    
    weekly_config = ReportConfig(
        report_type=ReportType.WEEKLY,
        format=ReportFormat.EXCEL,
        start_date=datetime.now() - timedelta(weeks=1),
        end_date=datetime.now(),
        recipients=config['report_recipients']['weekly'],
        schedule="09:00"
    )
    scheduler.schedule_report(weekly_config)
    
    # Run scheduler
    scheduler.run_scheduler()