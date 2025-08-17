#!/usr/bin/env python3
"""
ML Data Converter for Arbitrage Transactions
Converts JSON arbitrage data to various formats for machine learning training
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import h5py


class MLDataConverter:
    """Convert arbitrage JSON data to ML-ready formats"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = ['arb_type', 'dex_0', 'dex_1', 'dex_2']
        
    def load_json_data(self, filepath: str) -> List[Dict]:
        """Load arbitrage data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        print(f"Loaded {len(data)} arbitrage transactions")
        return data
    
    def flatten_transaction(self, tx: Dict) -> Dict:
        """Flatten nested transaction structure for ML processing"""
        flat = {
            # Identifiers
            'tx_signature': tx['tx_signature'],
            'slot': tx['slot'],
            'block_time': tx['block_time'],
            
            # Core metrics
            'revenue_sol': tx['revenue_sol'],
            'tx_fee_sol': tx['costs']['tx_fee_sol'],
            'priority_fee_sol': tx['costs']['priority_fee_sol'],
            'jito_tip_sol': tx['costs']['jito_tip_sol'],
            'total_cost_sol': tx['costs']['total_cost_sol'],
            'gross_profit_sol': tx['profit']['gross_sol'],
            'net_profit_sol': tx['profit']['net_sol'],
            'roi': tx['profit']['roi'],
            
            # Compute units
            'cu_used': tx['priority']['cu_used'],
            'cu_price_micro_lamports': tx['priority']['cu_price_micro_lamports'],
            'priority_lamports': tx['priority']['priority_lamports'],
            
            # Latency
            'latency_ms': tx['latency']['submit_to_land_ms'],
            
            # Classification
            'arb_type': tx['classification']['type'],
            'dex_count': tx['classification']['dex_count'],
            'amm_count': tx['classification']['amm_count'],
            
            # Market conditions
            'spread_bps': tx['market']['spread_bps'],
            'volatility_5s_bps': tx['market']['volatility_5s_bps'],
            
            # Risk metrics
            'sandwich_risk_bps': tx['risk']['sandwich_risk_bps'],
            'backrun_seen': int(tx['risk']['backrun_seen']),
            'honeypot_flag': int(tx['risk']['honeypot_flag']),
            'token_age_sec': tx['risk']['token_age_sec'],
            'ownership_concentration_pct': tx['risk']['ownership_concentration_pct'],
            'freeze_auth_present': int(tx['risk']['freeze_auth_present']),
            'mint_auth_present': int(tx['risk']['mint_auth_present']),
            
            # Signer reputation
            'signer_success_rate_7d': tx['signer_reputation']['success_rate_7d'],
            'signer_pnl_7d_sol': tx['signer_reputation']['pnl_7d_sol'],
            'signer_avg_roi_7d': tx['signer_reputation']['avg_roi_7d'],
            
            # Labels
            'confidence': tx['confidence'],
            'label_is_arb': tx['label_is_arb'],
            'target_net_profit_sol': tx['target_net_profit_sol'],
            'target_roi': tx['target_roi'],
        }
        
        # Add leg statistics
        legs = tx['legs']
        flat['num_legs'] = len(legs)
        
        # Aggregate slippage metrics
        fee_bps_list = [leg['slippage']['fee_bps'] for leg in legs]
        impact_bps_list = [leg['slippage']['price_impact_bps'] for leg in legs]
        liquidity_list = [leg['slippage']['liquidity_after'] for leg in legs]
        
        flat['total_fees_bps'] = sum(fee_bps_list)
        flat['avg_fees_bps'] = np.mean(fee_bps_list)
        flat['max_fees_bps'] = max(fee_bps_list)
        
        flat['avg_price_impact_bps'] = np.mean(impact_bps_list)
        flat['max_price_impact_bps'] = max(impact_bps_list)
        flat['min_price_impact_bps'] = min(impact_bps_list)
        
        flat['min_liquidity'] = min(liquidity_list)
        flat['max_liquidity'] = max(liquidity_list)
        flat['avg_liquidity'] = np.mean(liquidity_list)
        
        # Add DEX path (first 3 DEXs)
        path = tx['path']
        for i in range(3):
            if i < len(path):
                flat[f'dex_{i}'] = path[i]
            else:
                flat[f'dex_{i}'] = 'none'
        
        # Add first and last leg details
        if legs:
            first_leg = legs[0]
            flat['first_sell_amount'] = first_leg['sell_amount']
            flat['first_buy_amount'] = first_leg['buy_amount']
            flat['first_effective_price'] = first_leg['slippage']['effective_price']
            flat['first_price_impact'] = first_leg['slippage']['price_impact_bps']
            
            last_leg = legs[-1]
            flat['last_sell_amount'] = last_leg['sell_amount']
            flat['last_buy_amount'] = last_leg['buy_amount']
            flat['last_effective_price'] = last_leg['slippage']['effective_price']
            flat['last_price_impact'] = last_leg['slippage']['price_impact_bps']
        
        # Calculate derived features
        flat['profit_margin'] = flat['net_profit_sol'] / max(flat['revenue_sol'], 0.000001)
        flat['cost_ratio'] = flat['total_cost_sol'] / max(flat['revenue_sol'], 0.000001)
        flat['efficiency'] = flat['net_profit_sol'] / max(flat['latency_ms'], 1)
        flat['risk_adjusted_profit'] = flat['net_profit_sol'] / max(flat['sandwich_risk_bps'], 1)
        
        return flat
    
    def create_dataframe(self, transactions: List[Dict]) -> pd.DataFrame:
        """Convert transactions to pandas DataFrame"""
        flattened = [self.flatten_transaction(tx) for tx in transactions]
        df = pd.DataFrame(flattened)
        
        # Convert timestamp to datetime
        df['block_time'] = pd.to_datetime(df['block_time'], unit='s')
        
        # Add time-based features
        df['hour'] = df['block_time'].dt.hour
        df['day_of_week'] = df['block_time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Categorize latency
        df['latency_category'] = pd.cut(
            df['latency_ms'],
            bins=[0, 200, 400, 600, float('inf')],
            labels=['ultra_fast', 'fast', 'medium', 'slow']
        )
        
        # Categorize profit
        df['profit_category'] = pd.cut(
            df['net_profit_sol'],
            bins=[-float('inf'), 0, 0.01, 0.05, 0.1, float('inf')],
            labels=['loss', 'small', 'medium', 'large', 'huge']
        )
        
        print(f"Created DataFrame with shape: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for ML"""
        df_encoded = df.copy()
        
        # One-hot encode categorical columns
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                # Use LabelEncoder for ordinal encoding option
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].fillna('unknown'))
                self.label_encoders[col] = le
                
                # Also create one-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Encode latency category
        if 'latency_category' in df_encoded.columns:
            df_encoded['latency_cat_encoded'] = pd.Categorical(df_encoded['latency_category']).codes
        
        # Encode profit category
        if 'profit_category' in df_encoded.columns:
            df_encoded['profit_cat_encoded'] = pd.Categorical(df_encoded['profit_category']).codes
        
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = 'label_is_arb',
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML training
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (features, target)
        """
        if exclude_cols is None:
            exclude_cols = [
                'tx_signature', 'block_time', 'slot', 
                'arb_type', 'latency_category', 'profit_category',
                'dex_0', 'dex_1', 'dex_2'
            ]
        
        # Encode categorical variables
        df_encoded = self.encode_categorical(df)
        
        # Select numeric features
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and excluded columns
        feature_cols = [col for col in numeric_cols 
                       if col != target_col and col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        X = df_encoded[feature_cols]
        y = df_encoded[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2, 
                  val_size: float = 0.1,
                  random_state: int = 42) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training data)
            random_state: Random seed
            
        Returns:
            Dictionary with split data
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_val_scaled = self.scale_features(X_val, fit=False)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
    
    def save_ml_data(self, data: Dict[str, Any], output_dir: str = 'ml_data'):
        """Save processed ML data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save CSV files
        pd.DataFrame(data['X_train']).to_csv(output_path / 'X_train.csv', index=False)
        pd.DataFrame(data['X_val']).to_csv(output_path / 'X_val.csv', index=False)
        pd.DataFrame(data['X_test']).to_csv(output_path / 'X_test.csv', index=False)
        
        pd.Series(data['y_train']).to_csv(output_path / 'y_train.csv', index=False, header=['label'])
        pd.Series(data['y_val']).to_csv(output_path / 'y_val.csv', index=False, header=['label'])
        pd.Series(data['y_test']).to_csv(output_path / 'y_test.csv', index=False, header=['label'])
        
        # Save scaled numpy arrays
        np.save(output_path / 'X_train_scaled.npy', data['X_train_scaled'])
        np.save(output_path / 'X_val_scaled.npy', data['X_val_scaled'])
        np.save(output_path / 'X_test_scaled.npy', data['X_test_scaled'])
        
        # Save HDF5 format (efficient for large datasets)
        with h5py.File(output_path / 'ml_data.h5', 'w') as f:
            f.create_dataset('X_train_scaled', data=data['X_train_scaled'])
            f.create_dataset('X_val_scaled', data=data['X_val_scaled'])
            f.create_dataset('X_test_scaled', data=data['X_test_scaled'])
            f.create_dataset('y_train', data=data['y_train'].values)
            f.create_dataset('y_val', data=data['y_val'].values)
            f.create_dataset('y_test', data=data['y_test'].values)
        
        # Save preprocessing objects
        with open(output_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(data['scaler'], f)
        
        with open(output_path / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(data['label_encoders'], f)
        
        # Save feature columns
        with open(output_path / 'feature_columns.json', 'w') as f:
            json.dump(data['feature_columns'], f, indent=2)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'n_train': len(data['X_train']),
            'n_val': len(data['X_val']),
            'n_test': len(data['X_test']),
            'n_features': len(data['feature_columns']),
            'feature_columns': data['feature_columns']
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"ML data saved to {output_path}")
        print(f"  - Train: {len(data['X_train'])} samples")
        print(f"  - Val: {len(data['X_val'])} samples")
        print(f"  - Test: {len(data['X_test'])} samples")
        print(f"  - Features: {len(data['feature_columns'])}")
    
    def create_feature_importance_df(self, feature_importances: np.ndarray) -> pd.DataFrame:
        """Create DataFrame of feature importances"""
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importances
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df


def main():
    """Example usage of MLDataConverter"""
    
    # Initialize converter
    converter = MLDataConverter()
    
    # Load example data
    data_path = 'examples/arbitrage-examples.json'
    transactions = converter.load_json_data(data_path)
    
    # Create DataFrame
    df = converter.create_dataframe(transactions)
    
    # Save raw DataFrame
    df.to_csv('ml_data/arbitrage_data_raw.csv', index=False)
    print(f"Saved raw data to ml_data/arbitrage_data_raw.csv")
    
    # Prepare features
    X, y = converter.prepare_features(df)
    
    # Split and scale data
    ml_data = converter.split_data(X, y)
    
    # Save ML-ready data
    converter.save_ml_data(ml_data)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("ML Data Preparation Complete!")
    print("="*50)
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (arbitrage): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Number of features: {len(converter.feature_columns)}")
    print("\nTop 10 features:")
    for i, col in enumerate(converter.feature_columns[:10], 1):
        print(f"  {i}. {col}")
    
    print("\nData is ready for ML training!")
    print("Load with: pd.read_csv('ml_data/X_train.csv')")
    print("Or numpy: np.load('ml_data/X_train_scaled.npy')")
    print("Or HDF5: h5py.File('ml_data/ml_data.h5', 'r')")


if __name__ == "__main__":
    main()