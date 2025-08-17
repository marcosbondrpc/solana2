# Bundle Landing Rate Anomaly Detection Report

## Executive Summary

Statistical analysis of MEV bundle landing rates reveals significant anomalies in entity performance, with several operators achieving statistically improbable success rates.

## Key Findings

### Statistical Overview
- **Total Entities Analyzed**: 89
- **Anomalous Entities Detected**: 12 (13.5%)
- **Significance Level (α)**: 0.05
- **Multiple Testing Correction**: Benjamini-Hochberg applied

### Anomalous Entities

#### Entity: B91
- **Landing Rate**: 78.4% (Peer Mean: 52.3%)
- **Sample Size**: 18,420 bundles
- **T-Statistic**: 14.832
- **P-Value**: 0.000002 (Adjusted: 0.000018)
- **Effect Size (Cohen's d)**: 2.41 → LARGE effect size
- **95% CI**: [76.8%, 80.0%]
- **Decision DNA**: b91a7c3d5e9f2468

**Interpretation**: B91's landing rate is 26.1 percentage points above peer mean, representing a highly significant deviation (p < 0.001) with large effect size.

#### Entity: JIT
- **Landing Rate**: 72.1% (Peer Mean: 52.3%)
- **Sample Size**: 14,280 bundles
- **T-Statistic**: 10.247
- **P-Value**: 0.000031 (Adjusted: 0.000186)
- **Effect Size**: 1.89 → LARGE effect size
- **95% CI**: [70.2%, 74.0%]

#### Entity: ARB
- **Landing Rate**: 69.8% (Peer Mean: 52.3%)
- **Sample Size**: 9,840 bundles
- **T-Statistic**: 7.623
- **P-Value**: 0.000214 (Adjusted: 0.000856)
- **Effect Size**: 1.52 → LARGE effect size

### Top Performers by Landing Rate

1. **B91**: 78.4% (18,420 bundles) ⚠️ ANOMALOUS
2. **WHL**: 74.2% (3,210 bundles) ⚠️ ANOMALOUS  
3. **JIT**: 72.1% (14,280 bundles) ⚠️ ANOMALOUS
4. **PRO**: 70.3% (6,890 bundles) ⚠️ ANOMALOUS
5. **ARB**: 69.8% (9,840 bundles) ⚠️ ANOMALOUS
6. **MEV**: 68.4% (11,230 bundles) ✓ Normal
7. **BOT**: 66.7% (8,910 bundles) ✓ Normal
8. **TRD**: 64.2% (5,420 bundles) ✓ Normal

### Statistical Distribution

```
Landing Rate Distribution:
  Q1 (25th percentile): 41.2%
  Median (50th):        52.3%
  Q3 (75th percentile): 61.7%
  
  Outlier Threshold (Q3 + 1.5*IQR): 77.2%
  Entities above threshold: 2 (B91, WHL)
```

### Temporal Analysis

#### 24-Hour Window
- **Mean Landing Rate**: 52.3%
- **Std Deviation**: 14.7%
- **Skewness**: 0.82 (moderately right-skewed)
- **Kurtosis**: 0.34 (slightly peaked)

#### 7-Day Trend
- B91: Consistent 75-80% range (σ = 2.1%)
- JIT: Improving trend from 68% to 72%
- ARB: Stable around 69-70%

### Hypothesis Testing Results

**Null Hypothesis (H₀)**: Entity landing rate equals population mean
**Alternative (H₁)**: Entity landing rate differs from population mean

| Entity | H₀ Rejected | Confidence | Direction |
|--------|------------|------------|-----------|
| B91    | Yes        | 99.9998%   | Higher    |
| JIT    | Yes        | 99.997%    | Higher    |
| ARB    | Yes        | 99.979%    | Higher    |
| WHL    | Yes        | 99.8%      | Higher    |
| PRO    | Yes        | 99.2%      | Higher    |
| LOW    | Yes        | 98.4%      | Lower     |

### Peer Comparison Matrix

```
Relative Performance (vs peer mean):
  B91: +49.9% relative improvement
  JIT: +37.9% relative improvement
  ARB: +33.5% relative improvement
  WHL: +41.9% relative improvement
  PRO: +34.4% relative improvement
```

## Technical Indicators

### Possible Optimizations Detected

1. **Priority Fee Optimization**
   - Anomalous entities show 2.3x higher average priority fees
   - Correlation between fee and landing rate: r = 0.64

2. **Timing Precision**
   - Top performers submit within 50ms of slot boundaries
   - Standard deviation of submission time: <100ms

3. **Network Path Optimization**
   - Likely direct validator connections
   - Estimated latency advantage: 15-30ms

4. **Bundle Composition**
   - Optimal bundle size: 2-3 transactions
   - Strategic transaction ordering detected

## Risk Assessment

### Competitive Advantage Sources
- **Technical**: Superior infrastructure (40%)
- **Economic**: Better pricing models (30%)
- **Information**: Proprietary data feeds (20%)
- **Strategic**: Advanced algorithms (10%)

### Market Impact
- Anomalous entities control ~42% of successful bundles
- Revenue concentration increasing (HHI +120 points/month)
- Barrier to entry rising for new operators

## Recommendations

1. **Immediate Actions**
   - Flag B91, JIT, ARB for detailed behavioral analysis
   - Monitor for coordinated activity patterns
   - Track wallet rotation frequencies

2. **Detection Enhancements**
   - Implement real-time anomaly scoring
   - Add temporal clustering analysis
   - Cross-reference with latency profiles

3. **Monitoring Thresholds**
   - Alert: Landing rate >70% over 1000+ bundles
   - Warning: Effect size >1.5 (large deviation)
   - Investigation: P-value <0.001 after correction

## Methodology

- **Statistical Test**: One-sample t-test with pooled variance
- **Multiple Comparisons**: Benjamini-Hochberg FDR control
- **Effect Size**: Cohen's d for standardized differences
- **Confidence Intervals**: 95% using t-distribution
- **Minimum Sample Size**: 30 bundles per entity

---

*Report Generated: 2025-08-17T09:45:00Z*
*Analysis Window: 24 hours*
*Total Bundles Analyzed: 142,380*
*Decision DNA: 7f3a9b2c6d8e1045*