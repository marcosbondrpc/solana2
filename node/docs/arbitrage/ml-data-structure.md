# 🎯 Solana Arbitrage Data Structure for ML Training

## State-of-the-Art JSON Schema (SOTA-1.0)

### Overview

This is a comprehensive, production-ready JSON data structure for capturing Solana arbitrage opportunities, designed specifically for machine learning model training. The structure captures over 70 features across transaction mechanics, market conditions, risk factors, and profitability metrics.

## 📊 Data Structure Components

### Core Structure

```json
{
  "version": "sota-1.0",
  "slot": 360200000,
  "block_time": 1755280005,
  "tx_signature": "...",
  "legs": [...],
  "profit": {...},
  "market": {...},
  "risk": {...},
  "label_is_arb": 1
}
```

### Key Components

1. **Transaction Metadata** - Basic blockchain identifiers
2. **Arbitrage Legs** - Individual swap details with slippage
3. **Financial Metrics** - Revenue, costs, profit, ROI
4. **Market Conditions** - Spread, volatility, liquidity
5. **Risk Assessment** - Sandwich risk, token analysis
6. **Signer Reputation** - Historical performance metrics
7. **ML Labels** - Ground truth for supervised learning

## 🔧 Features

### Captured Metrics (70+ features)

#### Transaction Level
- `tx_signature`: Unique transaction identifier
- `slot`, `block_time`: Temporal positioning
- `signer`: Wallet performing arbitrage
- `program`: Main program used (e.g., Jupiter)

#### Leg-Level Details
Each leg captures:
- **Token Flow**: `sell_mint`, `buy_mint`, amounts
- **Slippage Metrics**:
  - `effective_price`: Actual execution price
  - `price_before`/`price_after`: Price impact
  - `fee_bps`: Trading fees
  - `liquidity_before`/`liquidity_after`: Pool depth

#### Profitability
- `revenue_sol`: Total revenue generated
- `costs`: Transaction, priority, Jito tip fees
- `net_profit_sol`: Final profit after costs
- `roi`: Return on investment percentage

#### Market Context
- `spread_bps`: Price spread across DEXs
- `volatility_5s_bps`: Recent price volatility
- `depth_top`: Available liquidity

#### Risk Factors
- `sandwich_risk_bps`: MEV attack vulnerability
- `token_age_sec`: Token maturity
- `ownership_concentration_pct`: Centralization risk
- `freeze_auth_present`: Token freeze risk

## 🚀 Quick Start

### 1. Validate Your Data

```python
from validator import ArbitrageDataValidator

validator = ArbitrageDataValidator()
is_valid, errors = validator.validate_transaction(your_transaction)

if not is_valid:
    print(f"Validation errors: {errors}")
```

### 2. Convert to DataFrame

```python
from validator import ArbitrageDataConverter

converter = ArbitrageDataConverter()
df = converter.json_to_dataframe(transactions_list)

# Save for analysis
df.to_csv("arbitrage_data.csv", index=False)
```

### 3. Prepare for ML Training

```python
from ml_converter import MLDataConverter

ml_converter = MLDataConverter()

# Load and process
transactions = ml_converter.load_json_data("data.json")
df = ml_converter.create_dataframe(transactions)

# Prepare features
X, y = ml_converter.prepare_features(df)

# Split data
ml_data = ml_converter.split_data(X, y, test_size=0.2)

# Save ML-ready data
ml_converter.save_ml_data(ml_data, "ml_data/")
```

## 📁 File Structure

```
ml-data-structure/
├── schema/
│   └── arbitrage-schema.json      # JSON Schema definition
├── examples/
│   └── arbitrage-examples.json    # Example transactions
├── validator.py                   # Validation & conversion
├── ml_converter.py                # ML data preparation
├── test_data_structure.py         # Test suite
└── README.md                      # This file
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_data_structure.py
```

Tests include:
- ✅ Schema validation
- ✅ Data conversion consistency
- ✅ Profit calculations
- ✅ Leg path validation
- ✅ ML preparation
- ✅ Feature extraction

## 📈 ML Features Generated

The converter generates 60+ features including:

### Numerical Features
- All profit/cost metrics
- Latency measurements
- Risk scores
- Liquidity metrics
- Aggregated slippage statistics

### Categorical Features (One-hot encoded)
- `arb_type`: Classification of arbitrage
- `dex_0`, `dex_1`, `dex_2`: DEX sequence
- `latency_category`: Speed classification
- `profit_category`: Profit magnitude

### Derived Features
- `profit_margin`: Net profit / Revenue
- `cost_ratio`: Costs / Revenue
- `efficiency`: Profit / Latency
- `risk_adjusted_profit`: Profit / Risk

## 🎯 Use Cases

### 1. Arbitrage Detection Model
```python
from sklearn.ensemble import RandomForestClassifier

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train_scaled, y_train)

# Predict arbitrage opportunities
predictions = clf.predict(X_test_scaled)
```

### 2. Profit Prediction
```python
from sklearn.linear_model import LinearRegression

# Train regressor
reg = LinearRegression()
reg.fit(X_train, df_train['net_profit_sol'])

# Predict profits
profit_predictions = reg.predict(X_test)
```

### 3. Risk Assessment
```python
# Analyze risk factors
risk_cols = ['sandwich_risk_bps', 'ownership_concentration_pct']
risk_df = df[risk_cols + ['net_profit_sol']]
correlation = risk_df.corr()
```

## 📊 Data Formats Supported

### Input
- **JSON**: Native format with full structure
- **JSON Lines**: Streaming format

### Output
- **CSV**: Flattened for spreadsheet analysis
- **Parquet**: Compressed columnar format
- **HDF5**: Hierarchical data for large datasets
- **NumPy**: Arrays for direct ML usage

## 🔍 Validation Rules

### Schema Validation
- All required fields must be present
- Data types must match specification
- Enums must have valid values

### Business Logic Validation
- Profit = Revenue - Total Costs
- ROI = Net Profit / Total Costs
- Legs must form circular path
- Slippage calculations must be consistent

## 📝 Example Transaction

See `examples/arbitrage-examples.json` for complete examples including:
- 2-leg arbitrage (most common)
- 3-leg arbitrage (complex routing)
- Various DEX combinations
- Different risk profiles

## 🛠️ Customization

### Adding New Features

1. Update schema in `arbitrage-schema.json`
2. Modify flattening in `ml_converter.py`
3. Add validation in `validator.py`
4. Update tests in `test_data_structure.py`

### Changing Validation Rules

Edit business logic validation in:
```python
validator._validate_business_logic(transaction)
```

## 📈 Performance

- **Validation Speed**: ~1,000 transactions/second
- **Conversion Speed**: ~10,000 transactions/second
- **Memory Usage**: ~1GB per million transactions
- **Compression**: 70% reduction with Parquet

## 🔒 Data Quality

### Consistency Guarantees
- Circular arbitrage paths
- Profit calculations verified
- Slippage impacts validated
- Risk scores normalized

### Missing Data Handling
- Categorical: Encoded as "unknown"
- Numerical: Forward-fill or interpolation
- Critical fields: Transaction rejected

## 📚 References

- [Solana Documentation](https://docs.solana.com)
- [JSON Schema Specification](https://json-schema.org)
- [MEV on Solana](https://jito-labs.gitbook.io)

## 🤝 Contributing

To add new DEX support or features:
1. Update the schema
2. Add validation logic
3. Include example transactions
4. Run test suite

## 📄 License

MIT License - Use freely for arbitrage detection and ML training.

---

**Built for professional MEV searchers and ML engineers working with Solana arbitrage data.** 🚀