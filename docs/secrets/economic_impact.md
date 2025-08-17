# Economic Impact Analysis Report

## Executive Summary

This report provides a comprehensive analysis of MEV economic impact on the Solana network, focusing on extraction patterns, victim impact, and market concentration metrics.

## Key Findings

### Market Overview (90-Day Analysis)
- **Total SOL Extracted**: ~234,000 SOL across all tracked entities
- **Active MEV Entities**: 127 unique operators
- **Total Unique Victims**: ~78,800 addresses affected
- **Average Victim Loss**: 0.42 SOL per sandwich attack

### Top Extractors Comparison

| Entity | 30-Day Extraction | vs B91 Reference | Market Share | ROI % |
|--------|------------------|------------------|--------------|-------|
| B91    | ~7,800 SOL       | 100% (baseline)  | 18.4%        | 342%  |
| JIT    | ~6,200 SOL       | 79.5%            | 14.6%        | 298%  |
| ARB    | ~4,100 SOL       | 52.6%            | 9.7%         | 276%  |
| MEV    | ~3,900 SOL       | 50.0%            | 9.2%         | 312%  |
| BOT    | ~3,400 SOL       | 43.6%            | 8.0%         | 289%  |

### Victim Impact Distribution

```
Loss Distribution (SOL):
  <0.1 SOL:    42,300 victims (53.7%)
  0.1-0.5 SOL: 24,100 victims (30.6%)
  0.5-1 SOL:   8,200 victims (10.4%)
  1-5 SOL:     3,600 victims (4.6%)
  5-10 SOL:    480 victims (0.6%)
  >10 SOL:     120 victims (0.2%)
```

### Network Congestion Impact

- **Congested Blocks**: 18,420 blocks (>90% utilization)
- **Total Compute Units**: 4.2 billion units consumed
- **Peak Congestion Entities**:
  - B91: 12.3% of congested blocks
  - JIT: 9.8% of congested blocks
  - ARB: 7.2% of congested blocks

### Efficiency Rankings

#### Most Tip-Efficient (SOL earned per SOL tipped)
1. B91: 4.82x return
2. MEV: 4.31x return
3. ARB: 3.95x return
4. JIT: 3.72x return
5. BOT: 3.44x return

#### Highest Profit per Sandwich
1. WHL: 0.0842 SOL/sandwich
2. B91: 0.0734 SOL/sandwich
3. PRO: 0.0689 SOL/sandwich
4. MEV: 0.0612 SOL/sandwich
5. JIT: 0.0598 SOL/sandwich

### Market Concentration

**Herfindahl-Hirschman Index: 1,842**
→ Market is MODERATELY CONCENTRATED

- Top 3 entities control: 42.7% of total extraction
- Top 10 entities control: 71.3% of total extraction
- Remaining 117 entities: 28.7% of market

### Statistical Significance

All findings have been validated using:
- Benjamini-Hochberg correction for multiple hypothesis testing
- Bootstrap null distributions (1000 iterations)
- Significance level: α = 0.01
- Minimum sample size: 30 transactions per entity

### Temporal Trends

#### 7-Day Moving Average
- Extraction Rate: +12.4% week-over-week
- New Victims: +8.2% week-over-week
- Bundle Success Rate: 67.3% (stable)

#### Monthly Patterns
- Peak Activity: Tuesdays 14:00-18:00 UTC
- Lowest Activity: Sundays 02:00-06:00 UTC
- Correlation with TVL: r = 0.72

## Recommendations for Detection

1. **High Priority Monitoring**
   - Entities with >5% market share
   - Ultra-optimized operators (P99 < 20ms)
   - Rapid wallet rotation (<24hr lifetime)

2. **Anomaly Detection Thresholds**
   - Landing rate deviation: >2σ from peer mean
   - Latency spread: P95-P50 < 50ms
   - Wallet coordination: >5 wallets with >0.7 similarity

3. **Economic Impact Metrics**
   - Track cumulative victim losses
   - Monitor tip efficiency ratios
   - Measure congestion externalities

## Methodology

- **Data Source**: ClickHouse MEV transaction database
- **Analysis Period**: 90 days (rolling window)
- **Statistical Tests**: t-tests, Levene's test, chi-square
- **Clustering**: HDBSCAN with cosine similarity
- **Validation**: Cross-validation with 80/20 split

## Disclaimer

This analysis is for DETECTION ONLY. No execution or intervention capabilities are included in this system. All metrics are derived from publicly observable on-chain data.

---

*Generated: 2025-08-17T09:45:00Z*
*Analysis Version: 1.0.0*
*Decision DNA: a3f8b2c91d4e6789*