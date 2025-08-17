"""
Executive KPI Dashboard for Arbitrage Infrastructure
Real-time business intelligence and decision support system
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from clickhouse_driver import Client as ClickHouseClient
import redis.asyncio as redis
from scipy import stats
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KPICalculator:
    """Calculates key performance indicators for executive dashboard"""
    
    def __init__(self, clickhouse_config: Dict[str, Any], redis_config: Dict[str, Any]):
        self.ch_client = ClickHouseClient(
            host=clickhouse_config['host'],
            port=clickhouse_config['port'],
            database=clickhouse_config['database']
        )
        self.redis_config = redis_config
        self.redis_client = None
        
    async def initialize(self):
        """Initialize async connections"""
        self.redis_client = await redis.from_url(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}"
        )
        
    def calculate_roi(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate return on investment metrics"""
        query = f"""
        SELECT 
            SUM(actual_profit) as total_profit,
            SUM(gas_cost) as total_gas,
            SUM(capital_used) as total_capital,
            COUNT(*) as total_trades,
            AVG(actual_profit / NULLIF(capital_used, 0)) as avg_roi_per_trade
        FROM arbitrage_opportunities
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
          AND success = 1
        """
        
        result = self.ch_client.execute(query)[0]
        
        total_profit = result[0] or 0
        total_gas = result[1] or 0
        total_capital = result[2] or 1  # Avoid division by zero
        total_trades = result[3] or 0
        avg_roi = result[4] or 0
        
        net_profit = total_profit - total_gas
        total_roi = (net_profit / total_capital) * 100 if total_capital > 0 else 0
        
        # Calculate infrastructure costs (simplified)
        days = (end_date - start_date).days or 1
        infrastructure_cost = days * 500  # $500/day estimated
        
        return {
            'total_roi': total_roi,
            'net_profit': net_profit,
            'gross_profit': total_profit,
            'gas_costs': total_gas,
            'infrastructure_costs': infrastructure_cost,
            'total_trades': total_trades,
            'avg_roi_per_trade': avg_roi * 100,
            'profit_after_costs': net_profit - infrastructure_cost,
            'capital_efficiency': (net_profit / total_capital * 365 / days) * 100 if total_capital > 0 else 0
        }
        
    def calculate_market_share(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Estimate market share in MEV space"""
        # Get our volume
        our_query = f"""
        SELECT 
            SUM(capital_used) as our_volume,
            COUNT(DISTINCT token_pair) as unique_pairs,
            COUNT(DISTINCT dex_a) + COUNT(DISTINCT dex_b) as unique_dexes
        FROM arbitrage_opportunities
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        """
        
        our_results = self.ch_client.execute(our_query)[0]
        our_volume = our_results[0] or 0
        
        # Get competitor volume (if tracked)
        competitor_query = f"""
        SELECT 
            SUM(volume) as competitor_volume,
            COUNT(DISTINCT competitor_address) as num_competitors
        FROM competitor_transactions
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        """
        
        try:
            competitor_results = self.ch_client.execute(competitor_query)[0]
            competitor_volume = competitor_results[0] or 0
            num_competitors = competitor_results[1] or 0
        except:
            competitor_volume = our_volume * 10  # Estimate if no competitor data
            num_competitors = 50
            
        total_volume = our_volume + competitor_volume
        market_share = (our_volume / total_volume * 100) if total_volume > 0 else 0
        
        return {
            'market_share_percent': market_share,
            'our_volume': our_volume,
            'total_market_volume': total_volume,
            'num_competitors': num_competitors,
            'unique_pairs': our_results[1],
            'unique_dexes': our_results[2],
            'market_position': self._calculate_market_position(market_share)
        }
        
    def _calculate_market_position(self, market_share: float) -> str:
        """Determine market position based on share"""
        if market_share > 20:
            return "Market Leader"
        elif market_share > 10:
            return "Top Tier"
        elif market_share > 5:
            return "Competitive"
        elif market_share > 1:
            return "Growing"
        else:
            return "Emerging"
            
    def calculate_risk_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        query = f"""
        SELECT 
            timestamp,
            actual_profit,
            capital_used,
            success,
            slippage,
            gas_cost
        FROM arbitrage_opportunities
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp
        """
        
        results = self.ch_client.execute(query)
        
        if not results:
            return {
                'value_at_risk_95': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'recovery_factor': 0
            }
            
        df = pd.DataFrame(results, columns=['timestamp', 'profit', 'capital', 'success', 'slippage', 'gas'])
        
        # Calculate returns
        df['returns'] = (df['profit'] - df['gas']) / df['capital'].replace(0, 1)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(df['returns'], 5) * 100
        
        # Maximum Drawdown
        cumulative = df['returns'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max.replace(0, 1)
        max_drawdown = abs(drawdown.min()) * 100
        
        # Sharpe Ratio (annualized)
        risk_free_rate = 0.02 / 365  # Daily risk-free rate
        excess_returns = df['returns'] - risk_free_rate
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(365)) if excess_returns.std() > 0 else 0
        
        # Win Rate
        win_rate = (df['success'].sum() / len(df) * 100) if len(df) > 0 else 0
        
        # Profit Factor
        profits = df[df['profit'] > 0]['profit'].sum()
        losses = abs(df[df['profit'] < 0]['profit'].sum())
        profit_factor = profits / losses if losses > 0 else float('inf')
        
        # Recovery Factor
        total_profit = df['profit'].sum() - df['gas'].sum()
        recovery_factor = total_profit / max_drawdown if max_drawdown > 0 else float('inf')
        
        return {
            'value_at_risk_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,
            'avg_slippage': df['slippage'].mean(),
            'max_slippage': df['slippage'].max()
        }
        
    def calculate_operational_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate operational excellence metrics"""
        query = f"""
        SELECT 
            AVG(execution_time_ms) as avg_execution_time,
            QUANTILE(0.95)(execution_time_ms) as p95_execution_time,
            QUANTILE(0.99)(execution_time_ms) as p99_execution_time,
            COUNT(*) as total_opportunities,
            SUM(success) as successful_trades,
            AVG(gas_cost) as avg_gas_cost,
            COUNT(DISTINCT DATE(timestamp)) as operating_days
        FROM arbitrage_opportunities
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        """
        
        result = self.ch_client.execute(query)[0]
        
        total_opportunities = result[3] or 1
        successful_trades = result[4] or 0
        operating_days = result[6] or 1
        
        return {
            'avg_execution_time': result[0] or 0,
            'p95_execution_time': result[1] or 0,
            'p99_execution_time': result[2] or 0,
            'total_opportunities': total_opportunities,
            'successful_trades': successful_trades,
            'success_rate': (successful_trades / total_opportunities * 100) if total_opportunities > 0 else 0,
            'avg_gas_cost': result[5] or 0,
            'opportunities_per_day': total_opportunities / operating_days,
            'trades_per_day': successful_trades / operating_days,
            'system_efficiency': self._calculate_system_efficiency(result)
        }
        
    def _calculate_system_efficiency(self, metrics: tuple) -> float:
        """Calculate overall system efficiency score (0-100)"""
        # Weighted scoring based on key metrics
        execution_score = max(0, 100 - (metrics[0] or 100) / 10)  # Lower is better
        success_rate = (metrics[4] or 0) / (metrics[3] or 1) * 100
        
        # Weighted average
        efficiency = (execution_score * 0.3 + success_rate * 0.7)
        
        return min(100, max(0, efficiency))
        
    def calculate_growth_metrics(self, current_period: tuple, previous_period: tuple) -> Dict[str, Any]:
        """Calculate growth and trend metrics"""
        # Current period metrics
        current_query = f"""
        SELECT 
            SUM(actual_profit) as profit,
            COUNT(*) as trades,
            SUM(capital_used) as volume,
            COUNT(DISTINCT token_pair) as unique_pairs
        FROM arbitrage_opportunities
        WHERE timestamp BETWEEN '{current_period[0]}' AND '{current_period[1]}'
          AND success = 1
        """
        
        current = self.ch_client.execute(current_query)[0]
        
        # Previous period metrics
        previous_query = f"""
        SELECT 
            SUM(actual_profit) as profit,
            COUNT(*) as trades,
            SUM(capital_used) as volume,
            COUNT(DISTINCT token_pair) as unique_pairs
        FROM arbitrage_opportunities
        WHERE timestamp BETWEEN '{previous_period[0]}' AND '{previous_period[1]}'
          AND success = 1
        """
        
        previous = self.ch_client.execute(previous_query)[0]
        
        # Calculate growth rates
        profit_growth = self._calculate_growth_rate(current[0], previous[0])
        trade_growth = self._calculate_growth_rate(current[1], previous[1])
        volume_growth = self._calculate_growth_rate(current[2], previous[2])
        pair_growth = self._calculate_growth_rate(current[3], previous[3])
        
        return {
            'profit_growth': profit_growth,
            'trade_growth': trade_growth,
            'volume_growth': volume_growth,
            'pair_expansion': pair_growth,
            'current_profit': current[0] or 0,
            'previous_profit': previous[0] or 0,
            'current_volume': current[2] or 0,
            'previous_volume': previous[2] or 0,
            'momentum_score': self._calculate_momentum_score(
                profit_growth, trade_growth, volume_growth
            )
        }
        
    def _calculate_growth_rate(self, current: float, previous: float) -> float:
        """Calculate percentage growth rate"""
        if previous == 0 or previous is None:
            return 100 if current > 0 else 0
        return ((current or 0) - previous) / previous * 100
        
    def _calculate_momentum_score(self, profit_growth: float, trade_growth: float, volume_growth: float) -> str:
        """Calculate business momentum score"""
        avg_growth = (profit_growth + trade_growth + volume_growth) / 3
        
        if avg_growth > 50:
            return "Accelerating"
        elif avg_growth > 20:
            return "Strong Growth"
        elif avg_growth > 0:
            return "Steady Growth"
        elif avg_growth > -20:
            return "Slowing"
        else:
            return "Declining"


class ExecutiveDashboard:
    """Interactive executive dashboard for business intelligence"""
    
    def __init__(self, kpi_calculator: KPICalculator):
        self.kpi_calculator = kpi_calculator
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Arbitrage Executive Dashboard"
        )
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Executive Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Date Range Selector
            dbc.Row([
                dbc.Col([
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now(),
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ], width=4),
                dbc.Col([
                    dbc.Button("Refresh", id="refresh-btn", color="primary", className="me-2"),
                    dbc.Button("Export PDF", id="export-pdf", color="secondary", className="me-2"),
                    dbc.Button("Export Excel", id="export-excel", color="secondary")
                ], width=8, className="text-end")
            ], className="mb-4"),
            
            # KPI Cards Row 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Net Profit", className="card-title"),
                            html.H2(id="net-profit", children="$0"),
                            html.P(id="profit-change", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("ROI", className="card-title"),
                            html.H2(id="roi", children="0%"),
                            html.P(id="roi-change", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Market Share", className="card-title"),
                            html.H2(id="market-share", children="0%"),
                            html.P(id="market-position", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Success Rate", className="card-title"),
                            html.H2(id="success-rate", children="0%"),
                            html.P(id="total-trades", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # KPI Cards Row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sharpe Ratio", className="card-title"),
                            html.H2(id="sharpe-ratio", children="0"),
                            html.P("Risk-Adjusted Returns", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Max Drawdown", className="card-title"),
                            html.H2(id="max-drawdown", children="0%"),
                            html.P("Risk Exposure", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Capital Efficiency", className="card-title"),
                            html.H2(id="capital-efficiency", children="0%"),
                            html.P("Annualized Returns", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Efficiency", className="card-title"),
                            html.H2(id="system-efficiency", children="0%"),
                            html.P(id="avg-latency", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Charts Row 1
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="profit-timeline")
                ], width=8),
                dbc.Col([
                    dcc.Graph(id="profit-breakdown")
                ], width=4)
            ], className="mb-4"),
            
            # Charts Row 2
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="strategy-performance")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="market-share-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Charts Row 3
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="risk-metrics")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="growth-metrics")
                ], width=6)
            ], className="mb-4"),
            
            # Detailed Metrics Table
            dbc.Row([
                dbc.Col([
                    html.H3("Detailed Performance Metrics"),
                    html.Div(id="metrics-table")
                ])
            ], className="mb-4"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            ),
            
            # Hidden div to store data
            html.Div(id='intermediate-value', style={'display': 'none'})
            
        ], fluid=True)
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('intermediate-value', 'children')],
            [Input('refresh-btn', 'n_clicks'),
             Input('interval-component', 'n_intervals')],
            [State('date-range', 'start_date'),
             State('date-range', 'end_date')]
        )
        def update_data(n_clicks, n_intervals, start_date, end_date):
            """Fetch and store all KPI data"""
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            # Calculate all KPIs
            roi_metrics = self.kpi_calculator.calculate_roi(start, end)
            market_metrics = self.kpi_calculator.calculate_market_share(start, end)
            risk_metrics = self.kpi_calculator.calculate_risk_metrics(start, end)
            operational_metrics = self.kpi_calculator.calculate_operational_metrics(start, end)
            
            # Calculate growth (compare to previous period)
            period_length = (end - start).days
            previous_start = start - timedelta(days=period_length)
            previous_end = start
            
            growth_metrics = self.kpi_calculator.calculate_growth_metrics(
                (start, end),
                (previous_start, previous_end)
            )
            
            data = {
                'roi': roi_metrics,
                'market': market_metrics,
                'risk': risk_metrics,
                'operational': operational_metrics,
                'growth': growth_metrics
            }
            
            return [json.dumps(data)]
            
        @self.app.callback(
            [Output('net-profit', 'children'),
             Output('profit-change', 'children'),
             Output('roi', 'children'),
             Output('roi-change', 'children'),
             Output('market-share', 'children'),
             Output('market-position', 'children'),
             Output('success-rate', 'children'),
             Output('total-trades', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('max-drawdown', 'children'),
             Output('capital-efficiency', 'children'),
             Output('system-efficiency', 'children'),
             Output('avg-latency', 'children')],
            [Input('intermediate-value', 'children')]
        )
        def update_kpi_cards(json_data):
            """Update KPI cards with latest data"""
            if not json_data:
                return ["$0", "", "0%", "", "0%", "", "0%", "", "0", "0%", "0%", "0%", ""]
                
            data = json.loads(json_data)
            
            # Format profit with growth indicator
            net_profit = f"${data['roi']['net_profit']:,.2f}"
            profit_growth = data['growth']['profit_growth']
            profit_change = f"{'↑' if profit_growth > 0 else '↓'} {abs(profit_growth):.1f}% vs prev period"
            
            # Format ROI
            roi = f"{data['roi']['total_roi']:.2f}%"
            roi_change = f"Capital Efficiency: {data['roi']['capital_efficiency']:.1f}% p.a."
            
            # Format market share
            market_share = f"{data['market']['market_share_percent']:.2f}%"
            market_position = data['market']['market_position']
            
            # Format success rate
            success_rate = f"{data['operational']['success_rate']:.1f}%"
            total_trades = f"{data['operational']['successful_trades']:,} successful trades"
            
            # Format risk metrics
            sharpe = f"{data['risk']['sharpe_ratio']:.2f}"
            max_dd = f"{data['risk']['max_drawdown']:.1f}%"
            
            # Format efficiency metrics
            capital_eff = f"{data['roi']['capital_efficiency']:.1f}%"
            system_eff = f"{data['operational']['system_efficiency']:.1f}%"
            avg_latency = f"Avg latency: {data['operational']['avg_execution_time']:.1f}ms"
            
            return [
                net_profit, profit_change,
                roi, roi_change,
                market_share, market_position,
                success_rate, total_trades,
                sharpe, max_dd,
                capital_eff, system_eff, avg_latency
            ]
            
        @self.app.callback(
            Output('profit-timeline', 'figure'),
            [Input('intermediate-value', 'children')],
            [State('date-range', 'start_date'),
             State('date-range', 'end_date')]
        )
        def update_profit_timeline(json_data, start_date, end_date):
            """Update profit timeline chart"""
            if not json_data:
                return go.Figure()
                
            # Query daily profit data
            query = f"""
            SELECT 
                DATE(timestamp) as date,
                SUM(actual_profit) as gross_profit,
                SUM(gas_cost) as gas_costs,
                SUM(actual_profit - gas_cost) as net_profit,
                COUNT(*) as trades
            FROM arbitrage_opportunities
            WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
              AND success = 1
            GROUP BY DATE(timestamp)
            ORDER BY date
            """
            
            results = self.kpi_calculator.ch_client.execute(query)
            
            if not results:
                return go.Figure()
                
            df = pd.DataFrame(results, columns=['date', 'gross_profit', 'gas_costs', 'net_profit', 'trades'])
            
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=('Daily P&L', 'Cumulative Profit'),
                vertical_spacing=0.1
            )
            
            # Daily P&L
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['net_profit'],
                    name='Net Profit',
                    marker_color=['green' if x > 0 else 'red' for x in df['net_profit']]
                ),
                row=1, col=1
            )
            
            # Cumulative profit
            cumulative = df['net_profit'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=cumulative,
                    mode='lines+markers',
                    name='Cumulative',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Profit & Loss Timeline',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        @self.app.callback(
            Output('profit-breakdown', 'figure'),
            [Input('intermediate-value', 'children')]
        )
        def update_profit_breakdown(json_data):
            """Update profit breakdown pie chart"""
            if not json_data:
                return go.Figure()
                
            data = json.loads(json_data)
            
            labels = ['Trading Profit', 'Gas Costs', 'Infrastructure']
            values = [
                data['roi']['gross_profit'],
                data['roi']['gas_costs'],
                data['roi']['infrastructure_costs']
            ]
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    marker=dict(colors=['green', 'orange', 'red'])
                )
            ])
            
            fig.update_layout(
                title='Cost Breakdown',
                height=400
            )
            
            return fig
            
        @self.app.callback(
            Output('strategy-performance', 'figure'),
            [Input('intermediate-value', 'children')],
            [State('date-range', 'start_date'),
             State('date-range', 'end_date')]
        )
        def update_strategy_performance(json_data, start_date, end_date):
            """Update strategy performance chart"""
            query = f"""
            SELECT 
                strategy_type,
                SUM(actual_profit) as total_profit,
                COUNT(*) as trades,
                AVG(actual_profit) as avg_profit,
                SUM(success) / COUNT(*) as success_rate
            FROM arbitrage_opportunities
            WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY strategy_type
            ORDER BY total_profit DESC
            """
            
            results = self.kpi_calculator.ch_client.execute(query)
            
            if not results:
                return go.Figure()
                
            df = pd.DataFrame(results, columns=['strategy', 'profit', 'trades', 'avg_profit', 'success_rate'])
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Profit by Strategy', 'Success Rate by Strategy'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            fig.add_trace(
                go.Bar(
                    x=df['strategy'],
                    y=df['profit'],
                    name='Total Profit',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df['strategy'],
                    y=df['success_rate'] * 100,
                    name='Success Rate',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Strategy Performance Analysis',
                height=400,
                showlegend=False
            )
            
            return fig
            
        @self.app.callback(
            Output('market-share-chart', 'figure'),
            [Input('intermediate-value', 'children')]
        )
        def update_market_share(json_data):
            """Update market share visualization"""
            if not json_data:
                return go.Figure()
                
            data = json.loads(json_data)
            market = data['market']
            
            # Create gauge chart for market share
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=market['market_share_percent'],
                title={'text': f"Market Share - {market['market_position']}"},
                delta={'reference': 5, 'relative': True},
                gauge={
                    'axis': {'range': [None, 30]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 15], 'color': "gray"},
                        {'range': [15, 30], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ))
            
            fig.update_layout(height=400)
            
            return fig
            
        @self.app.callback(
            Output('risk-metrics', 'figure'),
            [Input('intermediate-value', 'children')]
        )
        def update_risk_metrics(json_data):
            """Update risk metrics visualization"""
            if not json_data:
                return go.Figure()
                
            data = json.loads(json_data)
            risk = data['risk']
            
            metrics = ['VaR (95%)', 'Max Drawdown', 'Win Rate', 'Profit Factor']
            values = [
                abs(risk['value_at_risk_95']),
                risk['max_drawdown'],
                risk['win_rate'],
                min(risk['profit_factor'], 10) * 10  # Scale for visualization
            ]
            
            fig = go.Figure(data=[
                go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name='Risk Profile'
                )
            ])
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title='Risk Profile',
                height=400
            )
            
            return fig
            
        @self.app.callback(
            Output('growth-metrics', 'figure'),
            [Input('intermediate-value', 'children')]
        )
        def update_growth_metrics(json_data):
            """Update growth metrics visualization"""
            if not json_data:
                return go.Figure()
                
            data = json.loads(json_data)
            growth = data['growth']
            
            categories = ['Profit', 'Trades', 'Volume', 'Pairs']
            growth_rates = [
                growth['profit_growth'],
                growth['trade_growth'],
                growth['volume_growth'],
                growth['pair_expansion']
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=growth_rates,
                    marker_color=['green' if x > 0 else 'red' for x in growth_rates],
                    text=[f"{x:.1f}%" for x in growth_rates],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Growth Metrics - {growth['momentum_score']}",
                yaxis_title='Growth Rate (%)',
                height=400
            )
            
            return fig
            
        @self.app.callback(
            Output('metrics-table', 'children'),
            [Input('intermediate-value', 'children')]
        )
        def update_metrics_table(json_data):
            """Update detailed metrics table"""
            if not json_data:
                return html.Div()
                
            data = json.loads(json_data)
            
            # Prepare table data
            table_data = [
                {'Category': 'Financial', 'Metric': 'Net Profit', 'Value': f"${data['roi']['net_profit']:,.2f}"},
                {'Category': 'Financial', 'Metric': 'Gross Profit', 'Value': f"${data['roi']['gross_profit']:,.2f}"},
                {'Category': 'Financial', 'Metric': 'Gas Costs', 'Value': f"${data['roi']['gas_costs']:,.2f}"},
                {'Category': 'Financial', 'Metric': 'Infrastructure Costs', 'Value': f"${data['roi']['infrastructure_costs']:,.2f}"},
                {'Category': 'Financial', 'Metric': 'ROI', 'Value': f"{data['roi']['total_roi']:.2f}%"},
                {'Category': 'Financial', 'Metric': 'Capital Efficiency', 'Value': f"{data['roi']['capital_efficiency']:.1f}% p.a."},
                
                {'Category': 'Operational', 'Metric': 'Total Trades', 'Value': f"{data['operational']['total_opportunities']:,}"},
                {'Category': 'Operational', 'Metric': 'Successful Trades', 'Value': f"{data['operational']['successful_trades']:,}"},
                {'Category': 'Operational', 'Metric': 'Success Rate', 'Value': f"{data['operational']['success_rate']:.1f}%"},
                {'Category': 'Operational', 'Metric': 'Avg Execution Time', 'Value': f"{data['operational']['avg_execution_time']:.1f}ms"},
                {'Category': 'Operational', 'Metric': 'P95 Execution Time', 'Value': f"{data['operational']['p95_execution_time']:.1f}ms"},
                {'Category': 'Operational', 'Metric': 'System Efficiency', 'Value': f"{data['operational']['system_efficiency']:.1f}%"},
                
                {'Category': 'Market', 'Metric': 'Market Share', 'Value': f"{data['market']['market_share_percent']:.2f}%"},
                {'Category': 'Market', 'Metric': 'Market Position', 'Value': data['market']['market_position']},
                {'Category': 'Market', 'Metric': 'Trading Volume', 'Value': f"${data['market']['our_volume']:,.2f}"},
                {'Category': 'Market', 'Metric': 'Unique Pairs', 'Value': f"{data['market']['unique_pairs']:,}"},
                {'Category': 'Market', 'Metric': 'Active DEXes', 'Value': f"{data['market']['unique_dexes']:,}"},
                
                {'Category': 'Risk', 'Metric': 'Sharpe Ratio', 'Value': f"{data['risk']['sharpe_ratio']:.2f}"},
                {'Category': 'Risk', 'Metric': 'Max Drawdown', 'Value': f"{data['risk']['max_drawdown']:.1f}%"},
                {'Category': 'Risk', 'Metric': 'Value at Risk (95%)', 'Value': f"{data['risk']['value_at_risk_95']:.2f}%"},
                {'Category': 'Risk', 'Metric': 'Win Rate', 'Value': f"{data['risk']['win_rate']:.1f}%"},
                {'Category': 'Risk', 'Metric': 'Profit Factor', 'Value': f"{data['risk']['profit_factor']:.2f}"},
                
                {'Category': 'Growth', 'Metric': 'Profit Growth', 'Value': f"{data['growth']['profit_growth']:.1f}%"},
                {'Category': 'Growth', 'Metric': 'Volume Growth', 'Value': f"{data['growth']['volume_growth']:.1f}%"},
                {'Category': 'Growth', 'Metric': 'Momentum', 'Value': data['growth']['momentum_score']}
            ]
            
            return dash_table.DataTable(
                data=table_data,
                columns=[
                    {"name": "Category", "id": "Category"},
                    {"name": "Metric", "id": "Metric"},
                    {"name": "Value", "id": "Value"}
                ],
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Category'},
                        'fontWeight': 'bold'
                    }
                ],
                page_size=25
            )
            
    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


if __name__ == "__main__":
    # Configuration
    config = {
        'clickhouse': {
            'host': 'localhost',
            'port': 9000,
            'database': 'arbitrage'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379
        }
    }
    
    # Initialize KPI calculator
    kpi_calculator = KPICalculator(config['clickhouse'], config['redis'])
    
    # Create and run dashboard
    dashboard = ExecutiveDashboard(kpi_calculator)
    dashboard.run(debug=True, port=8050)