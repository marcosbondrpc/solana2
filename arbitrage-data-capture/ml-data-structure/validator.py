#!/usr/bin/env python3
"""
Arbitrage Data Structure Validator
Validates JSON data against the SOTA-1.0 schema for ML training
"""

import json
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np


class ArbitrageDataValidator:
    """Validator for arbitrage transaction data structure"""
    
    def __init__(self, schema_path: str = "schema/arbitrage-schema.json"):
        """Initialize validator with schema"""
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.validator = Draft7Validator(self.schema)
        self.validation_stats = {
            "total_validated": 0,
            "valid": 0,
            "invalid": 0,
            "errors": []
        }
    
    def _load_schema(self) -> Dict:
        """Load JSON schema from file"""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            return json.load(f)
    
    def validate_transaction(self, transaction: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single transaction against the schema
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            validate(instance=transaction, schema=self.schema)
            
            # Additional business logic validation
            errors.extend(self._validate_business_logic(transaction))
            
            if not errors:
                self.validation_stats["valid"] += 1
                return True, []
            
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            
        self.validation_stats["invalid"] += 1
        self.validation_stats["errors"].extend(errors)
        self.validation_stats["total_validated"] += 1
        
        return False, errors
    
    def _validate_business_logic(self, transaction: Dict) -> List[str]:
        """
        Validate business logic rules beyond schema validation
        """
        errors = []
        
        # Check profit calculations
        revenue = transaction.get("revenue_sol", 0)
        total_cost = transaction["costs"]["total_cost_sol"]
        expected_net = revenue - total_cost
        actual_net = transaction["profit"]["net_sol"]
        
        if abs(expected_net - actual_net) > 0.000001:  # Allow small floating point errors
            errors.append(f"Net profit mismatch: expected {expected_net}, got {actual_net}")
        
        # Check ROI calculation
        if total_cost > 0:
            expected_roi = (actual_net / total_cost)
            actual_roi = transaction["profit"]["roi"]
            if abs(expected_roi - actual_roi) > 0.0001:
                errors.append(f"ROI mismatch: expected {expected_roi}, got {actual_roi}")
        
        # Check legs consistency
        legs = transaction["legs"]
        if len(legs) < 2:
            errors.append("Arbitrage must have at least 2 legs")
        
        # Validate leg chain (output of one leg should match input of next)
        for i in range(len(legs) - 1):
            current_leg = legs[i]
            next_leg = legs[i + 1]
            
            if current_leg["buy_mint"] != next_leg["sell_mint"]:
                errors.append(f"Leg {i} output doesn't match leg {i+1} input")
        
        # Check if arbitrage is circular (ends where it starts)
        if legs and legs[0]["sell_mint"] != legs[-1]["buy_mint"]:
            errors.append("Arbitrage path is not circular")
        
        # Validate slippage values
        for i, leg in enumerate(legs):
            slippage = leg["slippage"]
            if slippage["price_before"] <= 0 or slippage["price_after"] <= 0:
                errors.append(f"Leg {i}: Invalid price values")
            
            # Check price impact calculation
            expected_impact = ((slippage["price_after"] - slippage["price_before"]) / 
                             slippage["price_before"]) * 10000
            actual_impact = slippage["price_impact_bps"]
            
            if abs(expected_impact - actual_impact) > 1:  # Allow 1 bps tolerance
                errors.append(f"Leg {i}: Price impact calculation mismatch")
        
        # Validate confidence score
        confidence = transaction.get("confidence", 0)
        label = transaction.get("label_is_arb", 0)
        
        if label == 1 and confidence < 0.5:
            errors.append("High confidence expected for positive arbitrage label")
        
        # Validate latency
        latency_ms = transaction["latency"]["submit_to_land_ms"]
        if latency_ms < 0 or latency_ms > 60000:  # Max 60 seconds
            errors.append(f"Unrealistic latency value: {latency_ms}ms")
        
        # Validate signer reputation values
        reputation = transaction["signer_reputation"]
        if not 0 <= reputation["success_rate_7d"] <= 1:
            errors.append("Success rate must be between 0 and 1")
        
        return errors
    
    def validate_batch(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Validate a batch of transactions
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Validation report dictionary
        """
        results = []
        
        for i, tx in enumerate(transactions):
            is_valid, errors = self.validate_transaction(tx)
            results.append({
                "index": i,
                "tx_signature": tx.get("tx_signature", "unknown"),
                "is_valid": is_valid,
                "errors": errors
            })
        
        return {
            "total": len(transactions),
            "valid": sum(1 for r in results if r["is_valid"]),
            "invalid": sum(1 for r in results if not r["is_valid"]),
            "results": results,
            "stats": self.validation_stats
        }
    
    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """
        Validate transactions from a JSON file
        
        Args:
            filepath: Path to JSON file containing transactions
            
        Returns:
            Validation report
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return self.validate_batch(data)
        else:
            is_valid, errors = self.validate_transaction(data)
            return {
                "total": 1,
                "valid": 1 if is_valid else 0,
                "invalid": 0 if is_valid else 1,
                "errors": errors
            }
    
    def generate_validation_report(self) -> str:
        """Generate a human-readable validation report"""
        stats = self.validation_stats
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║              ARBITRAGE DATA VALIDATION REPORT                      ║
╚════════════════════════════════════════════════════════════════════╝

Total Transactions Validated: {stats['total_validated']}
Valid Transactions: {stats['valid']} ({stats['valid']/max(stats['total_validated'], 1)*100:.1f}%)
Invalid Transactions: {stats['invalid']} ({stats['invalid']/max(stats['total_validated'], 1)*100:.1f}%)

"""
        
        if stats['errors']:
            report += "Common Errors:\n"
            error_counts = {}
            for error in stats['errors']:
                error_type = error.split(':')[0]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                report += f"  - {error_type}: {count} occurrences\n"
        
        return report


class ArbitrageDataConverter:
    """Convert arbitrage JSON data to formats suitable for ML training"""
    
    @staticmethod
    def json_to_dataframe(transactions: List[Dict]) -> pd.DataFrame:
        """
        Convert list of arbitrage transactions to pandas DataFrame
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Flattened DataFrame suitable for ML training
        """
        # Flatten nested structures
        flattened_data = []
        
        for tx in transactions:
            flat_tx = {
                # Basic fields
                "version": tx["version"],
                "slot": tx["slot"],
                "block_time": tx["block_time"],
                "tx_signature": tx["tx_signature"],
                "signer": tx["signer"],
                "program": tx["program"],
                
                # Revenue and costs
                "revenue_sol": tx["revenue_sol"],
                "tx_fee_sol": tx["costs"]["tx_fee_sol"],
                "priority_fee_sol": tx["costs"]["priority_fee_sol"],
                "jito_tip_sol": tx["costs"]["jito_tip_sol"],
                "total_cost_sol": tx["costs"]["total_cost_sol"],
                
                # Profit metrics
                "gross_profit_sol": tx["profit"]["gross_sol"],
                "net_profit_sol": tx["profit"]["net_sol"],
                "roi": tx["profit"]["roi"],
                
                # Priority
                "cu_used": tx["priority"]["cu_used"],
                "cu_price_micro_lamports": tx["priority"]["cu_price_micro_lamports"],
                "priority_lamports": tx["priority"]["priority_lamports"],
                
                # Latency
                "latency_ms": tx["latency"]["submit_to_land_ms"],
                
                # Classification
                "arb_type": tx["classification"]["type"],
                "dex_count": tx["classification"]["dex_count"],
                "amm_count": tx["classification"]["amm_count"],
                
                # Market metrics
                "spread_bps": tx["market"]["spread_bps"],
                "volatility_5s_bps": tx["market"]["volatility_5s_bps"],
                
                # Risk metrics
                "sandwich_risk_bps": tx["risk"]["sandwich_risk_bps"],
                "backrun_seen": int(tx["risk"]["backrun_seen"]),
                "honeypot_flag": int(tx["risk"]["honeypot_flag"]),
                "token_age_sec": tx["risk"]["token_age_sec"],
                "ownership_concentration_pct": tx["risk"]["ownership_concentration_pct"],
                "freeze_auth_present": int(tx["risk"]["freeze_auth_present"]),
                "mint_auth_present": int(tx["risk"]["mint_auth_present"]),
                
                # Signer reputation
                "signer_success_rate_7d": tx["signer_reputation"]["success_rate_7d"],
                "signer_pnl_7d_sol": tx["signer_reputation"]["pnl_7d_sol"],
                "signer_avg_roi_7d": tx["signer_reputation"]["avg_roi_7d"],
                
                # ML labels
                "confidence": tx["confidence"],
                "label_is_arb": tx["label_is_arb"],
                "target_net_profit_sol": tx["target_net_profit_sol"],
                "target_roi": tx["target_roi"],
                
                # Aggregate leg statistics
                "num_legs": len(tx["legs"]),
                "total_fees_bps": sum(leg["slippage"]["fee_bps"] for leg in tx["legs"]),
                "avg_price_impact_bps": np.mean([leg["slippage"]["price_impact_bps"] for leg in tx["legs"]]),
                "max_price_impact_bps": max(leg["slippage"]["price_impact_bps"] for leg in tx["legs"]),
                "min_liquidity": min(leg["slippage"]["liquidity_after"] for leg in tx["legs"]),
            }
            
            # Add path as categorical features
            for i, dex in enumerate(tx["path"][:3]):  # Limit to first 3 DEXs
                flat_tx[f"dex_{i}"] = dex
            
            # Add first and last leg details
            if tx["legs"]:
                first_leg = tx["legs"][0]
                flat_tx["first_sell_amount"] = first_leg["sell_amount"]
                flat_tx["first_effective_price"] = first_leg["slippage"]["effective_price"]
                
                last_leg = tx["legs"][-1]
                flat_tx["last_buy_amount"] = last_leg["buy_amount"]
                flat_tx["last_effective_price"] = last_leg["slippage"]["effective_price"]
            
            flattened_data.append(flat_tx)
        
        df = pd.DataFrame(flattened_data)
        
        # Convert timestamp to datetime
        df['block_time'] = pd.to_datetime(df['block_time'], unit='s')
        
        # Add derived features useful for ML
        df['profit_margin'] = df['net_profit_sol'] / df['revenue_sol'].replace(0, 1)
        df['fee_ratio'] = df['total_cost_sol'] / df['revenue_sol'].replace(0, 1)
        df['latency_category'] = pd.cut(df['latency_ms'], 
                                       bins=[0, 100, 500, 1000, float('inf')],
                                       labels=['ultra_fast', 'fast', 'medium', 'slow'])
        
        return df
    
    @staticmethod
    def dataframe_to_csv(df: pd.DataFrame, filepath: str):
        """Save DataFrame to CSV for ML training"""
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")
    
    @staticmethod
    def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for ML training
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Select features for training
        feature_columns = [
            'revenue_sol', 'total_cost_sol', 'roi', 'cu_used',
            'latency_ms', 'dex_count', 'amm_count', 'spread_bps',
            'volatility_5s_bps', 'sandwich_risk_bps', 'token_age_sec',
            'ownership_concentration_pct', 'signer_success_rate_7d',
            'signer_avg_roi_7d', 'num_legs', 'total_fees_bps',
            'avg_price_impact_bps', 'max_price_impact_bps', 'min_liquidity',
            'first_sell_amount', 'first_effective_price',
            'last_buy_amount', 'last_effective_price'
        ]
        
        # Handle categorical variables
        categorical_columns = ['arb_type', 'dex_0', 'dex_1', 'dex_2']
        
        # Create feature matrix
        X = df[feature_columns].copy()
        
        # One-hot encode categorical variables if they exist
        for col in categorical_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                X = pd.concat([X, dummies], axis=1)
        
        # Label for supervised learning
        y = df['label_is_arb']
        
        return X, y


if __name__ == "__main__":
    # Example usage
    validator = ArbitrageDataValidator()
    
    # Test with example data
    example_tx = {
        "version": "sota-1.0",
        "slot": 360200000,
        "block_time": 1755280005,
        "tx_signature": "3JupRxM2k7y1r5oJp8xLk2aUqf7Dq7Tq2sJvB7E7jHhFJm4ZVYpJ1m3J1pZ4r1ke3A1Y2p7D8dJfQJ7h9XwQJtq",
        "signer": "GtagyESa99t49VmUqnnfsuowYnigSNKuYXdXWyXWNdd",
        "program": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        "path": ["Raydium CPMM", "Meteora DLMM"],
        "legs": [
            {
                "inner_program": "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",
                "amm": "Raydium CPMM",
                "pool_id": "raydium_cpmmpool_pepe_sol",
                "sell_mint": "So11111111111111111111111111111111111111112",
                "buy_mint": "EkJuyYyD3to61CHVPJn6wHb7xANxvqApnVJ4o2SdBAGS",
                "sell_amount": 2.5,
                "buy_amount": 201000.0,
                "slippage": {
                    "effective_price": 80400.0,
                    "price_before": 80320.0,
                    "price_after": 80410.0,
                    "fee_bps": 25.0,
                    "price_impact_bps": 11.2,
                    "liquidity_before": 12000000.0,
                    "liquidity_after": 11875000.0
                }
            },
            {
                "inner_program": "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",
                "amm": "Meteora DLMM",
                "pool_id": "meteora_dlmm_pepe_sol",
                "sell_mint": "EkJuyYyD3to61CHVPJn6wHb7xANxvqApnVJ4o2SdBAGS",
                "buy_mint": "So11111111111111111111111111111111111111112",
                "sell_amount": 201000.0,
                "buy_amount": 2.535,
                "slippage": {
                    "effective_price": 79369.0,
                    "price_before": 79250.0,
                    "price_after": 79380.0,
                    "fee_bps": 30.0,
                    "price_impact_bps": 16.4,
                    "liquidity_before": 8800000.0,
                    "liquidity_after": 8855000.0
                }
            }
        ],
        "revenue_sol": 0.035,
        "costs": {
            "tx_fee_sol": 0.00001,
            "priority_fee_sol": 0.00012,
            "jito_tip_sol": 0.0,
            "total_cost_sol": 0.00013
        },
        "profit": {
            "gross_sol": 0.035,
            "net_sol": 0.03487,
            "roi": 268.23
        },
        "priority": {
            "cu_used": 340000,
            "cu_price_micro_lamports": 350,
            "priority_lamports": 119.0
        },
        "latency": {
            "submit_to_land_ms": 380
        },
        "classification": {
            "type": "dex<->dex_two_leg",
            "dex_count": 2,
            "amm_count": 2
        },
        "market": {
            "snapshot_id": "sol-pepe-360200000",
            "mid_across_dex": {
                "Raydium": 0.00001245,
                "Meteora": 0.0000126
            },
            "depth_top": {
                "Raydium": 11500000,
                "Meteora": 9000000
            },
            "spread_bps": 7.1,
            "volatility_5s_bps": 18.3
        },
        "risk": {
            "sandwich_risk_bps": 1.1,
            "backrun_seen": False,
            "honeypot_flag": False,
            "token_age_sec": 864000,
            "ownership_concentration_pct": 7.4,
            "freeze_auth_present": False,
            "mint_auth_present": False
        },
        "signer_reputation": {
            "success_rate_7d": 0.82,
            "pnl_7d_sol": 412.7,
            "avg_roi_7d": 0.0091
        },
        "confidence": 0.96,
        "label_is_arb": 1,
        "target_net_profit_sol": 0.03487,
        "target_roi": 268.23
    }
    
    # Validate example
    is_valid, errors = validator.validate_transaction(example_tx)
    print(f"Validation result: {'VALID' if is_valid else 'INVALID'}")
    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  - {error}")
    
    # Convert to DataFrame
    converter = ArbitrageDataConverter()
    df = converter.json_to_dataframe([example_tx])
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")