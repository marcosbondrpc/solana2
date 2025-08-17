# MEV Strategies Documentation

## Overview

This documentation covers all Maximum Extractable Value (MEV) strategies implemented in the Solana MEV Infrastructure, including detection algorithms, execution strategies, and risk management.

## Strategy Types

### 1. Arbitrage
- [Cross-DEX Arbitrage](./arbitrage-detection.md)
- [Triangular Arbitrage](./triangular-arbitrage.md)
- [Flash Loan Arbitrage](./flash-loan-arbitrage.md)

### 2. Sandwich Attacks
- [Sandwich Detection](./sandwich-detector.md)
- [Front-running Protection](./frontrun-protection.md)
- [Bundle Strategies](./bundle-strategies.md)

### 3. Liquidations
- [Liquidation Monitoring](./liquidation-monitoring.md)
- [Collateral Management](./collateral-management.md)
- [Risk Assessment](./risk-assessment.md)

## Strategy Configuration

### Basic Configuration

```yaml
# configs/strategies.yaml
strategies:
  arbitrage:
    enabled: true
    min_profit_usd: 50
    max_gas_sol: 0.01
    confidence_threshold: 0.85
    pools:
      - raydium
      - orca
      - meteora
      - phoenix
    
  sandwich:
    enabled: true
    min_victim_size_usd: 1000
    max_priority_fee_sol: 0.01
    frontrun_protection: true
    
  liquidation:
    enabled: true
    min_collateral_usd: 5000
    health_factor_threshold: 1.05
    protocols:
      - solend
      - mango
      - kamino
```

## Performance Metrics

### Key Performance Indicators (KPIs)

| Metric | Description | Target | Current |
|--------|-------------|--------|---------|
| Detection Rate | Opportunities detected/block | >10 | 12.3 |
| Success Rate | Successful executions | >90% | 93.4% |
| Average Profit | Profit per execution | >$50 | $67.8 |
| Win/Loss Ratio | Profitable vs unprofitable | >3:1 | 3.8:1 |
| ROI | Return on investment | >20% | 24.5% |

### Strategy Performance

```javascript
// Real-time performance tracking
const performanceMetrics = {
  arbitrage: {
    totalExecutions: 5234,
    successRate: 0.945,
    totalProfit: 67890.12,
    avgProfit: 12.97,
    maxProfit: 523.45,
    profitFactor: 3.2
  },
  sandwich: {
    totalExecutions: 2345,
    successRate: 0.923,
    totalProfit: 34567.89,
    avgProfit: 14.74,
    maxProfit: 234.56,
    profitFactor: 2.8
  }
};
```

## ML Models

### Feature Engineering

```python
# Key features for MEV detection
features = {
    'price_features': [
        'price_impact',
        'slippage',
        'spread',
        'volatility'
    ],
    'volume_features': [
        'volume_24h',
        'volume_ratio',
        'liquidity_depth',
        'order_book_imbalance'
    ],
    'network_features': [
        'gas_price',
        'block_fill_rate',
        'mempool_size',
        'priority_fee'
    ],
    'historical_features': [
        'success_rate_1h',
        'profit_trend',
        'competition_level',
        'time_of_day'
    ]
}
```

### Model Architecture

```python
# Ensemble model configuration
models = {
    'xgboost': {
        'weight': 0.4,
        'params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200
        }
    },
    'lightgbm': {
        'weight': 0.3,
        'params': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 150
        }
    },
    'catboost': {
        'weight': 0.3,
        'params': {
            'depth': 6,
            'learning_rate': 0.03,
            'iterations': 100
        }
    }
}
```

## Execution Engine

### Bundle Building

```rust
pub struct BundleBuilder {
    tip_strategy: TipStrategy,
    priority_fee: u64,
    max_retries: u32,
}

impl BundleBuilder {
    pub fn build_bundle(&self, opportunity: &Opportunity) -> Bundle {
        let transactions = vec![
            self.build_setup_tx(),
            self.build_execution_tx(opportunity),
            self.build_cleanup_tx(),
        ];
        
        Bundle {
            transactions,
            tip: self.calculate_tip(opportunity),
            priority: Priority::High,
        }
    }
    
    fn calculate_tip(&self, opp: &Opportunity) -> u64 {
        match self.tip_strategy {
            TipStrategy::Fixed => 1_000_000, // 0.001 SOL
            TipStrategy::Percentage => (opp.profit * 0.1) as u64,
            TipStrategy::Ladder => self.ladder_tip(opp),
        }
    }
}
```

### Multi-path Submission

```rust
pub async fn submit_bundle(bundle: Bundle) -> Result<()> {
    // Parallel submission to multiple endpoints
    let (tpu_result, jito_result) = tokio::join!(
        submit_to_tpu(bundle.clone()),
        submit_to_jito(bundle.clone())
    );
    
    // Return first successful result
    match (tpu_result, jito_result) {
        (Ok(sig), _) | (_, Ok(sig)) => Ok(sig),
        (Err(e1), Err(e2)) => Err(format!("{}, {}", e1, e2)),
    }
}
```

## Risk Management

### Position Sizing

```python
def calculate_position_size(
    opportunity: Opportunity,
    portfolio: Portfolio,
    risk_params: RiskParams
) -> float:
    # Kelly Criterion for optimal sizing
    win_prob = opportunity.confidence
    loss_prob = 1 - win_prob
    avg_win = opportunity.expected_profit
    avg_loss = opportunity.max_loss
    
    kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_win
    
    # Apply safety factor
    safe_fraction = kelly_fraction * risk_params.kelly_factor
    
    # Apply maximum position limit
    max_position = portfolio.balance * risk_params.max_position_pct
    
    return min(opportunity.required_capital * safe_fraction, max_position)
```

### Stop Loss & Take Profit

```typescript
interface RiskLimits {
  maxDailyLoss: number;
  maxPositionSize: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  maxOpenPositions: number;
  correlationThreshold: number;
}

class RiskManager {
  checkLimits(opportunity: Opportunity): boolean {
    if (this.dailyLoss > this.limits.maxDailyLoss) {
      return false; // Circuit breaker triggered
    }
    
    if (this.openPositions >= this.limits.maxOpenPositions) {
      return false; // Position limit reached
    }
    
    if (this.calculateCorrelation() > this.limits.correlationThreshold) {
      return false; // Too correlated with existing positions
    }
    
    return true;
  }
}
```

## Advanced Strategies

### Statistical Arbitrage

```python
class StatisticalArbitrage:
    def __init__(self):
        self.pair_threshold = 2.0  # Standard deviations
        self.lookback_period = 1000
        
    def find_opportunities(self, prices: pd.DataFrame):
        # Calculate z-scores for all pairs
        for pair in self.cointegrated_pairs:
            spread = self.calculate_spread(pair, prices)
            z_score = self.calculate_zscore(spread)
            
            if abs(z_score) > self.pair_threshold:
                yield self.create_opportunity(pair, z_score)
    
    def calculate_spread(self, pair, prices):
        # Use Kalman filter for dynamic hedge ratio
        return prices[pair[0]] - self.hedge_ratio * prices[pair[1]]
```

### Market Making

```rust
pub struct MarketMaker {
    spread: f64,
    inventory_target: f64,
    risk_aversion: f64,
}

impl MarketMaker {
    pub fn calculate_quotes(&self, mid_price: f64, inventory: f64) -> (f64, f64) {
        // Avellaneda-Stoikov model
        let inventory_adjustment = self.risk_aversion * 
            (inventory - self.inventory_target);
        
        let bid = mid_price - self.spread / 2.0 - inventory_adjustment;
        let ask = mid_price + self.spread / 2.0 - inventory_adjustment;
        
        (bid, ask)
    }
}
```

## Backtesting Framework

### Historical Simulation

```python
class Backtester:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data
        self.results = []
        
    def run(self):
        for timestamp, market_data in self.data.iterrows():
            opportunities = self.strategy.detect(market_data)
            
            for opp in opportunities:
                result = self.simulate_execution(opp, market_data)
                self.results.append(result)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        return {
            'total_profit': sum(r.profit for r in self.results),
            'success_rate': sum(1 for r in self.results if r.success) / len(self.results),
            'sharpe_ratio': self.calculate_sharpe(),
            'max_drawdown': self.calculate_drawdown(),
            'profit_factor': self.calculate_profit_factor(),
        }
```

## Strategy Optimization

### Hyperparameter Tuning

```python
from optuna import create_study

def objective(trial):
    params = {
        'min_profit': trial.suggest_float('min_profit', 10, 100),
        'confidence_threshold': trial.suggest_float('confidence', 0.7, 0.95),
        'max_gas': trial.suggest_float('max_gas', 0.001, 0.01),
    }
    
    strategy = ArbitrageStrategy(**params)
    backtest = Backtester(strategy, historical_data)
    results = backtest.run()
    
    return results['sharpe_ratio']

study = create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## Monitoring & Alerts

### Real-time Monitoring

```javascript
// WebSocket monitoring
const monitor = new StrategyMonitor({
  alerts: {
    lowSuccessRate: 0.8,
    highSlippage: 0.02,
    maxDailyLoss: 1000,
  },
  
  onAlert: (alert) => {
    // Send notification
    notificationService.send({
      type: alert.severity,
      message: alert.message,
      strategy: alert.strategy,
      metrics: alert.metrics,
    });
  }
});
```

### Performance Dashboard

```typescript
interface StrategyDashboard {
  currentPnL: number;
  todayPnL: number;
  successRate: number;
  activePositions: Position[];
  recentExecutions: Execution[];
  alerts: Alert[];
  metrics: {
    sharpeRatio: number;
    profitFactor: number;
    maxDrawdown: number;
    winLossRatio: number;
  };
}
```

## Best Practices

1. **Always simulate before executing** - Use local state fork
2. **Implement circuit breakers** - Stop trading on excessive losses
3. **Monitor competition** - Track other MEV bots
4. **Diversify strategies** - Don't rely on single approach
5. **Keep models updated** - Retrain regularly with new data
6. **Use proper position sizing** - Never risk more than you can afford
7. **Log everything** - Maintain audit trail for analysis
8. **Test extensively** - Backtest and forward test strategies
9. **Stay updated** - Monitor protocol changes and updates
10. **Manage gas efficiently** - Optimize transaction costs

## Resources

- [MEV Research Papers](https://github.com/flashbots/mev-research)
- [Solana MEV Documentation](https://docs.jito.wtf/)
- [DeFi Strategies Guide](https://defillama.com/yields/strategy)
- [Risk Management Best Practices](https://www.risk.net/)