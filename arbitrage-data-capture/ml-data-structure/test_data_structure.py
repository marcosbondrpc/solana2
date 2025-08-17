#!/usr/bin/env python3
"""
Test Suite for Arbitrage Data Structure
Validates consistency, conversion, and ML readiness
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from validator import ArbitrageDataValidator, ArbitrageDataConverter
from ml_converter import MLDataConverter


def test_schema_validation():
    """Test schema validation with examples"""
    print("\n" + "="*60)
    print("TESTING SCHEMA VALIDATION")
    print("="*60)
    
    validator = ArbitrageDataValidator()
    
    # Load example transactions
    with open('examples/arbitrage-examples.json', 'r') as f:
        examples = json.load(f)
    
    print(f"Testing {len(examples)} example transactions...")
    
    all_valid = True
    for i, tx in enumerate(examples):
        is_valid, errors = validator.validate_transaction(tx)
        
        if is_valid:
            print(f"  ✓ Transaction {i+1} ({tx['tx_signature'][:8]}...): VALID")
        else:
            print(f"  ✗ Transaction {i+1} ({tx['tx_signature'][:8]}...): INVALID")
            for error in errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
            all_valid = False
    
    # Print validation report
    report = validator.generate_validation_report()
    print(report)
    
    return all_valid


def test_data_conversion():
    """Test conversion to DataFrame"""
    print("\n" + "="*60)
    print("TESTING DATA CONVERSION")
    print("="*60)
    
    converter = ArbitrageDataConverter()
    
    # Load examples
    with open('examples/arbitrage-examples.json', 'r') as f:
        examples = json.load(f)
    
    # Convert to DataFrame
    df = converter.json_to_dataframe(examples)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)[:10]}...")
    
    # Check data types
    print("\nData types summary:")
    print(df.dtypes.value_counts())
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values found in columns:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df[['net_profit_sol', 'roi', 'latency_ms', 'confidence']].describe())
    
    return df


def test_ml_preparation():
    """Test ML data preparation"""
    print("\n" + "="*60)
    print("TESTING ML DATA PREPARATION")
    print("="*60)
    
    ml_converter = MLDataConverter()
    
    # Load and convert data
    transactions = ml_converter.load_json_data('examples/arbitrage-examples.json')
    df = ml_converter.create_dataframe(transactions)
    
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Prepare features
    X, y = ml_converter.prepare_features(df)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Test train/val/test split
    ml_data = ml_converter.split_data(X, y, test_size=0.3, val_size=0.2)
    
    print(f"\nData split:")
    print(f"  Train: {ml_data['X_train'].shape}")
    print(f"  Val: {ml_data['X_val'].shape}")
    print(f"  Test: {ml_data['X_test'].shape}")
    
    # Check scaling
    print(f"\nScaled data stats:")
    print(f"  Train mean: {ml_data['X_train_scaled'].mean():.6f}")
    print(f"  Train std: {ml_data['X_train_scaled'].std():.6f}")
    
    return ml_data


def test_consistency():
    """Test consistency across conversions"""
    print("\n" + "="*60)
    print("TESTING DATA CONSISTENCY")
    print("="*60)
    
    # Load original data
    with open('examples/arbitrage-examples.json', 'r') as f:
        original = json.load(f)
    
    # Convert using validator converter
    validator_converter = ArbitrageDataConverter()
    df1 = validator_converter.json_to_dataframe(original)
    
    # Convert using ML converter
    ml_converter = MLDataConverter()
    df2 = ml_converter.create_dataframe(original)
    
    print(f"Validator DataFrame shape: {df1.shape}")
    print(f"ML Converter DataFrame shape: {df2.shape}")
    
    # Check key columns consistency
    key_columns = ['net_profit_sol', 'roi', 'latency_ms', 'label_is_arb']
    
    consistent = True
    for col in key_columns:
        if col in df1.columns and col in df2.columns:
            if df1[col].equals(df2[col]):
                print(f"  ✓ Column '{col}' is consistent")
            else:
                print(f"  ✗ Column '{col}' has differences")
                consistent = False
        else:
            print(f"  ⚠ Column '{col}' missing in one DataFrame")
    
    return consistent


def test_profit_calculations():
    """Test profit and ROI calculations"""
    print("\n" + "="*60)
    print("TESTING PROFIT CALCULATIONS")
    print("="*60)
    
    with open('examples/arbitrage-examples.json', 'r') as f:
        examples = json.load(f)
    
    all_correct = True
    for i, tx in enumerate(examples):
        revenue = tx['revenue_sol']
        total_cost = tx['costs']['total_cost_sol']
        net_profit = tx['profit']['net_sol']
        roi = tx['profit']['roi']
        
        # Calculate expected values
        expected_net = revenue - total_cost
        expected_roi = (expected_net / total_cost) if total_cost > 0 else 0
        
        # Check calculations (with small tolerance for floating point)
        net_diff = abs(expected_net - net_profit)
        roi_diff = abs(expected_roi - roi)
        
        if net_diff < 0.000001:
            print(f"  ✓ Tx {i+1}: Net profit correct ({net_profit:.6f} SOL)")
        else:
            print(f"  ✗ Tx {i+1}: Net profit mismatch (expected {expected_net:.6f}, got {net_profit:.6f})")
            all_correct = False
        
        if roi_diff < 0.01:
            print(f"  ✓ Tx {i+1}: ROI correct ({roi:.2f})")
        else:
            print(f"  ✗ Tx {i+1}: ROI mismatch (expected {expected_roi:.2f}, got {roi:.2f})")
            all_correct = False
    
    return all_correct


def test_leg_consistency():
    """Test arbitrage leg path consistency"""
    print("\n" + "="*60)
    print("TESTING LEG PATH CONSISTENCY")
    print("="*60)
    
    with open('examples/arbitrage-examples.json', 'r') as f:
        examples = json.load(f)
    
    all_valid = True
    for i, tx in enumerate(examples):
        legs = tx['legs']
        
        print(f"\nTransaction {i+1} ({len(legs)} legs):")
        
        # Check if path is circular
        first_sell = legs[0]['sell_mint']
        last_buy = legs[-1]['buy_mint']
        
        if first_sell == last_buy:
            print(f"  ✓ Circular path (starts and ends with same token)")
        else:
            print(f"  ✗ Non-circular path")
            all_valid = False
        
        # Check leg connections
        for j in range(len(legs) - 1):
            current_buy = legs[j]['buy_mint']
            next_sell = legs[j+1]['sell_mint']
            
            if current_buy == next_sell:
                print(f"  ✓ Leg {j+1} → Leg {j+2} connected")
            else:
                print(f"  ✗ Leg {j+1} → Leg {j+2} disconnected")
                all_valid = False
    
    return all_valid


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("ARBITRAGE DATA STRUCTURE TEST REPORT")
    print("="*60)
    
    results = {
        'schema_validation': test_schema_validation(),
        'data_conversion': test_data_conversion() is not None,
        'ml_preparation': test_ml_preparation() is not None,
        'consistency': test_consistency(),
        'profit_calculations': test_profit_calculations(),
        'leg_consistency': test_leg_consistency()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("Data structure is consistent and ML-ready.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please review the errors above.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    # Run all tests
    all_passed = generate_test_report()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)