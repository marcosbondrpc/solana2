"""
Data Quality Validation Pipeline
Comprehensive data validation, anomaly detection, and quality scoring
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import great_expectations as ge
from great_expectations.core import ExpectationConfiguration
from prometheus_client import Counter, Gauge, Histogram
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Prometheus metrics
data_validation_total = Counter('data_validation_total', 'Total data validations', ['dataset', 'status'])
data_quality_score = Gauge('data_quality_score', 'Data quality score (0-100)', ['dataset'])
validation_duration = Histogram('validation_duration_seconds', 'Validation duration', ['dataset'])
data_anomalies_detected = Counter('data_anomalies_detected_total', 'Data anomalies detected', ['dataset', 'type'])
duplicate_records = Counter('duplicate_records_total', 'Duplicate records detected', ['dataset'])
missing_values = Counter('missing_values_total', 'Missing values detected', ['dataset', 'field'])
schema_violations = Counter('schema_violations_total', 'Schema violations detected', ['dataset', 'field'])

class DataQualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # 95-100
    GOOD = "good"           # 85-94
    ACCEPTABLE = "acceptable"  # 70-84
    POOR = "poor"           # 50-69
    UNACCEPTABLE = "unacceptable"  # <50

class ValidationStatus(Enum):
    """Validation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class AnomalyType(Enum):
    """Types of data anomalies"""
    OUTLIER = "outlier"
    DUPLICATE = "duplicate"
    MISSING = "missing"
    SCHEMA_VIOLATION = "schema_violation"
    CONSISTENCY = "consistency"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"

@dataclass
class ValidationRule:
    """Data validation rule"""
    name: str
    field: str
    rule_type: str  # range, regex, enum, custom
    parameters: Dict[str, Any]
    severity: str = "warning"  # info, warning, error, critical
    enabled: bool = True
    description: str = ""

@dataclass
class ValidationResult:
    """Result of data validation"""
    dataset: str
    timestamp: datetime
    status: ValidationStatus
    quality_score: float
    total_records: int
    valid_records: int
    invalid_records: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    field_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_duration_seconds: float = 0

@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics"""
    completeness: float  # Percentage of non-null values
    consistency: float   # Percentage of consistent records
    accuracy: float      # Percentage of accurate values
    timeliness: float    # Data freshness score
    uniqueness: float    # Percentage of unique records
    validity: float      # Percentage of valid values
    
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        return (
            self.completeness * 0.25 +
            self.consistency * 0.20 +
            self.accuracy * 0.20 +
            self.timeliness * 0.15 +
            self.uniqueness * 0.10 +
            self.validity * 0.10
        )

class DataQualityValidator:
    """
    Comprehensive data quality validation system with
    anomaly detection and quality scoring
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.expectations: Dict[str, ge.dataset.Dataset] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.validation_history: Dict[str, List[ValidationResult]] = defaultdict(list)
        
        # Statistical baselines for anomaly detection
        self.statistical_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Data lineage tracking
        self.data_lineage: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize default validation rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default validation rules for different datasets"""
        
        # Arbitrage transaction rules
        self.validation_rules['arbitrage_transactions'] = [
            ValidationRule(
                name="profit_range",
                field="net_profit",
                rule_type="range",
                parameters={"min": -1000000, "max": 100000000},
                severity="error",
                description="Net profit must be within reasonable range"
            ),
            ValidationRule(
                name="roi_percentage",
                field="roi_percentage",
                rule_type="range",
                parameters={"min": -100, "max": 10000},
                severity="warning",
                description="ROI percentage must be within bounds"
            ),
            ValidationRule(
                name="gas_cost_positive",
                field="gas_cost",
                rule_type="range",
                parameters={"min": 0, "max": 10000000},
                severity="error",
                description="Gas cost must be positive"
            ),
            ValidationRule(
                name="dex_count",
                field="dex_count",
                rule_type="range",
                parameters={"min": 1, "max": 10},
                severity="warning",
                description="DEX count must be reasonable"
            ),
            ValidationRule(
                name="timestamp_valid",
                field="block_timestamp",
                rule_type="custom",
                parameters={"function": "validate_timestamp"},
                severity="error",
                description="Timestamp must be valid and recent"
            )
        ]
        
        # Market snapshot rules
        self.validation_rules['market_snapshots'] = [
            ValidationRule(
                name="price_positive",
                field="price",
                rule_type="range",
                parameters={"min": 0, "max": float('inf')},
                severity="error",
                description="Price must be positive"
            ),
            ValidationRule(
                name="liquidity_positive",
                field="total_liquidity",
                rule_type="range",
                parameters={"min": 0, "max": float('inf')},
                severity="error",
                description="Liquidity must be positive"
            ),
            ValidationRule(
                name="spread_reasonable",
                field="spread_bps",
                rule_type="range",
                parameters={"min": 0, "max": 10000},
                severity="warning",
                description="Spread must be reasonable"
            )
        ]
        
        # Risk metrics rules
        self.validation_rules['risk_metrics'] = [
            ValidationRule(
                name="risk_score_range",
                field="overall_risk_level",
                rule_type="enum",
                parameters={"values": ["low", "medium", "high", "extreme"]},
                severity="error",
                description="Risk level must be valid"
            ),
            ValidationRule(
                name="probability_range",
                field="frontrun_probability",
                rule_type="range",
                parameters={"min": 0, "max": 1},
                severity="error",
                description="Probability must be between 0 and 1"
            )
        ]
    
    async def validate_dataset(self, dataset_name: str, 
                              data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate a dataset and return quality metrics"""
        
        start_time = datetime.utcnow()
        result = ValidationResult(
            dataset=dataset_name,
            timestamp=start_time,
            status=ValidationStatus.PASSED,
            quality_score=100.0,
            total_records=len(data),
            valid_records=len(data),
            invalid_records=0
        )
        
        if not data:
            result.warnings.append("Empty dataset")
            result.status = ValidationStatus.WARNING
            return result
        
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # 1. Schema validation
            schema_issues = await self._validate_schema(dataset_name, df)
            if schema_issues:
                result.warnings.extend(schema_issues)
            
            # 2. Apply validation rules
            rule_violations = await self._apply_validation_rules(dataset_name, df)
            if rule_violations:
                result.errors.extend(rule_violations)
                result.invalid_records = len(rule_violations)
            
            # 3. Check for duplicates
            duplicates = await self._check_duplicates(dataset_name, df)
            if duplicates:
                result.warnings.append(f"Found {duplicates} duplicate records")
                duplicate_records.labels(dataset=dataset_name).inc(duplicates)
            
            # 4. Check for missing values
            missing_stats = await self._check_missing_values(dataset_name, df)
            result.field_stats['missing_values'] = missing_stats
            
            # 5. Detect anomalies
            anomalies = await self._detect_anomalies(dataset_name, df)
            if anomalies:
                result.anomalies = anomalies
                for anomaly in anomalies:
                    data_anomalies_detected.labels(
                        dataset=dataset_name,
                        type=anomaly['type']
                    ).inc()
            
            # 6. Check data consistency
            consistency_issues = await self._check_consistency(dataset_name, df)
            if consistency_issues:
                result.warnings.extend(consistency_issues)
            
            # 7. Calculate data quality metrics
            quality_metrics = await self._calculate_quality_metrics(dataset_name, df)
            result.quality_score = quality_metrics.overall_score()
            result.field_stats['quality_metrics'] = {
                'completeness': quality_metrics.completeness,
                'consistency': quality_metrics.consistency,
                'accuracy': quality_metrics.accuracy,
                'timeliness': quality_metrics.timeliness,
                'uniqueness': quality_metrics.uniqueness,
                'validity': quality_metrics.validity
            }
            
            # 8. Statistical analysis
            stats = await self._calculate_statistics(df)
            result.field_stats['statistics'] = stats
            
            # 9. Update baselines for future comparisons
            await self._update_baselines(dataset_name, df)
            
            # Determine final status
            if result.errors:
                result.status = ValidationStatus.FAILED
            elif result.warnings:
                result.status = ValidationStatus.WARNING
            
            # Determine quality level
            quality_level = self._determine_quality_level(result.quality_score)
            result.field_stats['quality_level'] = quality_level.value
            
            # Calculate duration
            result.validation_duration_seconds = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            # Update metrics
            data_validation_total.labels(
                dataset=dataset_name,
                status=result.status.value
            ).inc()
            data_quality_score.labels(dataset=dataset_name).set(result.quality_score)
            validation_duration.labels(dataset=dataset_name).observe(
                result.validation_duration_seconds
            )
            
            # Store in history
            self.validation_history[dataset_name].append(result)
            
            # Trim history to last 100 validations
            if len(self.validation_history[dataset_name]) > 100:
                self.validation_history[dataset_name] = (
                    self.validation_history[dataset_name][-100:]
                )
            
            logger.info(
                f"Validation completed for {dataset_name}",
                quality_score=result.quality_score,
                status=result.status.value,
                duration=result.validation_duration_seconds
            )
            
        except Exception as e:
            logger.error(f"Validation error for {dataset_name}: {e}")
            result.status = ValidationStatus.FAILED
            result.errors.append(f"Validation error: {str(e)}")
            result.quality_score = 0
        
        return result
    
    async def _validate_schema(self, dataset_name: str, df: pd.DataFrame) -> List[str]:
        """Validate data schema"""
        issues = []
        
        # Expected schemas for different datasets
        expected_schemas = {
            'arbitrage_transactions': {
                'signature': 'object',
                'block_height': 'int64',
                'net_profit': 'int64',
                'roi_percentage': 'float64',
                'dex_count': 'int64'
            },
            'market_snapshots': {
                'pool_address': 'object',
                'price': 'float64',
                'total_liquidity': 'int64',
                'spread_bps': 'int64'
            }
        }
        
        if dataset_name in expected_schemas:
            expected = expected_schemas[dataset_name]
            
            for column, expected_type in expected.items():
                if column not in df.columns:
                    issues.append(f"Missing required column: {column}")
                    schema_violations.labels(dataset=dataset_name, field=column).inc()
                elif str(df[column].dtype) != expected_type:
                    actual_type = str(df[column].dtype)
                    issues.append(
                        f"Schema violation: {column} expected {expected_type}, got {actual_type}"
                    )
                    schema_violations.labels(dataset=dataset_name, field=column).inc()
        
        return issues
    
    async def _apply_validation_rules(self, dataset_name: str, 
                                     df: pd.DataFrame) -> List[str]:
        """Apply validation rules to dataset"""
        violations = []
        
        if dataset_name not in self.validation_rules:
            return violations
        
        rules = self.validation_rules[dataset_name]
        
        for rule in rules:
            if not rule.enabled:
                continue
            
            if rule.field not in df.columns:
                continue
            
            try:
                if rule.rule_type == "range":
                    min_val = rule.parameters.get("min", float('-inf'))
                    max_val = rule.parameters.get("max", float('inf'))
                    
                    invalid = df[
                        (df[rule.field] < min_val) | (df[rule.field] > max_val)
                    ]
                    
                    if not invalid.empty:
                        violations.append(
                            f"{rule.name}: {len(invalid)} records violate range [{min_val}, {max_val}]"
                        )
                
                elif rule.rule_type == "enum":
                    valid_values = rule.parameters.get("values", [])
                    invalid = df[~df[rule.field].isin(valid_values)]
                    
                    if not invalid.empty:
                        violations.append(
                            f"{rule.name}: {len(invalid)} records have invalid values"
                        )
                
                elif rule.rule_type == "regex":
                    pattern = rule.parameters.get("pattern", "")
                    if pattern:
                        invalid = df[~df[rule.field].str.match(pattern)]
                        if not invalid.empty:
                            violations.append(
                                f"{rule.name}: {len(invalid)} records don't match pattern"
                            )
                
                elif rule.rule_type == "custom":
                    func_name = rule.parameters.get("function")
                    if func_name == "validate_timestamp":
                        # Check if timestamps are recent (within last 7 days)
                        now = pd.Timestamp.now()
                        week_ago = now - pd.Timedelta(days=7)
                        
                        if pd.api.types.is_datetime64_any_dtype(df[rule.field]):
                            invalid = df[
                                (df[rule.field] < week_ago) | (df[rule.field] > now)
                            ]
                            if not invalid.empty:
                                violations.append(
                                    f"{rule.name}: {len(invalid)} records have invalid timestamps"
                                )
                
            except Exception as e:
                logger.error(f"Error applying rule {rule.name}: {e}")
        
        return violations
    
    async def _check_duplicates(self, dataset_name: str, df: pd.DataFrame) -> int:
        """Check for duplicate records"""
        
        # Define key columns for different datasets
        key_columns = {
            'arbitrage_transactions': ['signature'],
            'market_snapshots': ['pool_address', 'snapshot_time'],
            'risk_metrics': ['transaction_signature']
        }
        
        if dataset_name in key_columns:
            keys = key_columns[dataset_name]
            keys = [k for k in keys if k in df.columns]
            
            if keys:
                duplicates = df.duplicated(subset=keys, keep='first')
                return duplicates.sum()
        
        # Fallback to checking all columns
        return df.duplicated().sum()
    
    async def _check_missing_values(self, dataset_name: str, 
                                   df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in dataset"""
        missing_stats = {}
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                missing_stats[column] = {
                    'count': int(missing_count),
                    'percentage': round(missing_percentage, 2)
                }
                missing_values.labels(dataset=dataset_name, field=column).inc(missing_count)
        
        return missing_stats
    
    async def _detect_anomalies(self, dataset_name: str, 
                               df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in the dataset"""
        anomalies = []
        
        # Numerical columns for anomaly detection
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_columns:
            return anomalies
        
        try:
            # Prepare features
            features = df[numerical_columns].fillna(0).values
            
            # Initialize or get anomaly detector
            if dataset_name not in self.anomaly_detectors:
                self.anomaly_detectors[dataset_name] = IsolationForest(
                    contamination=0.05,  # Expect 5% anomalies
                    random_state=42
                )
                self.scalers[dataset_name] = StandardScaler()
                
                # Fit the model
                if len(features) > 10:
                    scaler = self.scalers[dataset_name]
                    features_scaled = scaler.fit_transform(features)
                    self.anomaly_detectors[dataset_name].fit(features_scaled)
            
            # Detect anomalies
            if dataset_name in self.anomaly_detectors and len(features) > 0:
                scaler = self.scalers[dataset_name]
                features_scaled = scaler.transform(features)
                
                predictions = self.anomaly_detectors[dataset_name].predict(features_scaled)
                anomaly_scores = self.anomaly_detectors[dataset_name].score_samples(features_scaled)
                
                # Find anomalous records
                anomaly_indices = np.where(predictions == -1)[0]
                
                for idx in anomaly_indices[:10]:  # Limit to top 10 anomalies
                    anomaly_record = df.iloc[idx].to_dict()
                    
                    # Determine anomaly type
                    anomaly_type = AnomalyType.OUTLIER.value
                    
                    # Check for specific anomaly patterns
                    if 'profit' in anomaly_record:
                        profit = anomaly_record.get('net_profit', 0)
                        if profit > df['net_profit'].quantile(0.99):
                            anomaly_type = "extreme_profit"
                    
                    anomalies.append({
                        'type': anomaly_type,
                        'record_index': int(idx),
                        'anomaly_score': float(anomaly_scores[idx]),
                        'details': {k: v for k, v in anomaly_record.items() 
                                  if k in numerical_columns}
                    })
            
            # Statistical anomaly detection
            for column in numerical_columns:
                if column in df.columns:
                    # Z-score based anomaly detection
                    z_scores = np.abs(stats.zscore(df[column].dropna()))
                    threshold = 3  # 3 standard deviations
                    
                    statistical_anomalies = np.where(z_scores > threshold)[0]
                    
                    if len(statistical_anomalies) > 0:
                        for idx in statistical_anomalies[:5]:
                            anomalies.append({
                                'type': AnomalyType.STATISTICAL.value,
                                'field': column,
                                'value': float(df[column].iloc[idx]),
                                'z_score': float(z_scores[idx]),
                                'threshold': threshold
                            })
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    async def _check_consistency(self, dataset_name: str, 
                                df: pd.DataFrame) -> List[str]:
        """Check data consistency"""
        issues = []
        
        # Dataset-specific consistency checks
        if dataset_name == 'arbitrage_transactions':
            # Check if net_profit = profit_amount - gas_cost
            if all(col in df.columns for col in ['net_profit', 'profit_amount', 'gas_cost']):
                calculated_net = df['profit_amount'] - df['gas_cost']
                inconsistent = df[abs(df['net_profit'] - calculated_net) > 1]
                
                if not inconsistent.empty:
                    issues.append(
                        f"Profit calculation inconsistency in {len(inconsistent)} records"
                    )
            
            # Check if ROI calculation is correct
            if all(col in df.columns for col in ['roi_percentage', 'net_profit', 'gas_cost']):
                mask = df['gas_cost'] > 0
                calculated_roi = (df.loc[mask, 'net_profit'] / df.loc[mask, 'gas_cost']) * 100
                inconsistent = df.loc[mask][
                    abs(df.loc[mask, 'roi_percentage'] - calculated_roi) > 0.01
                ]
                
                if not inconsistent.empty:
                    issues.append(
                        f"ROI calculation inconsistency in {len(inconsistent)} records"
                    )
        
        elif dataset_name == 'market_snapshots':
            # Check if bid/ask spread is consistent
            if all(col in df.columns for col in ['spread_bps', 'price']):
                # Spread should be positive
                negative_spread = df[df['spread_bps'] < 0]
                if not negative_spread.empty:
                    issues.append(
                        f"Negative spread in {len(negative_spread)} records"
                    )
        
        # Temporal consistency
        if 'timestamp' in df.columns or 'block_timestamp' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'block_timestamp'
            
            # Check for future timestamps
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                future_records = df[df[time_col] > pd.Timestamp.now()]
                if not future_records.empty:
                    issues.append(
                        f"Future timestamps in {len(future_records)} records"
                    )
            
            # Check for chronological order issues
            if len(df) > 1:
                time_diffs = df[time_col].diff()
                backwards = time_diffs[time_diffs < pd.Timedelta(0)]
                if not backwards.empty:
                    issues.append(
                        f"Non-chronological order detected in {len(backwards)} records"
                    )
        
        return issues
    
    async def _calculate_quality_metrics(self, dataset_name: str, 
                                        df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics"""
        
        total_cells = df.size
        total_records = len(df)
        
        # Completeness: percentage of non-null values
        non_null_cells = df.notna().sum().sum()
        completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Uniqueness: percentage of unique records
        duplicate_count = df.duplicated().sum()
        uniqueness = ((total_records - duplicate_count) / total_records * 100) if total_records > 0 else 100
        
        # Validity: percentage of records passing validation rules
        validity = 100.0  # Default, will be adjusted based on validation results
        if dataset_name in self.validation_rules:
            # Simple validity check based on rules
            valid_count = total_records
            for rule in self.validation_rules[dataset_name]:
                if rule.field in df.columns and rule.rule_type == "range":
                    min_val = rule.parameters.get("min", float('-inf'))
                    max_val = rule.parameters.get("max", float('inf'))
                    invalid = df[(df[rule.field] < min_val) | (df[rule.field] > max_val)]
                    valid_count -= len(invalid)
            
            validity = (valid_count / total_records * 100) if total_records > 0 else 0
        
        # Consistency: based on consistency checks
        consistency = 95.0  # Default high consistency, reduced by issues
        
        # Accuracy: difficult to measure without ground truth, use proxy metrics
        accuracy = 90.0  # Default, adjusted based on anomalies
        
        # Timeliness: data freshness
        timeliness = 100.0
        if 'timestamp' in df.columns or 'block_timestamp' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'block_timestamp'
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                latest = df[time_col].max()
                age_hours = (pd.Timestamp.now() - latest).total_seconds() / 3600
                
                if age_hours < 1:
                    timeliness = 100.0
                elif age_hours < 24:
                    timeliness = 90.0
                elif age_hours < 168:  # 1 week
                    timeliness = 70.0
                else:
                    timeliness = 50.0
        
        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            uniqueness=uniqueness,
            validity=validity
        )
    
    async def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical summary of the dataset"""
        stats = {
            'record_count': len(df),
            'column_count': len(df.columns),
            'numerical_columns': {},
            'categorical_columns': {}
        }
        
        # Numerical column statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in df.columns:
                stats['numerical_columns'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75)),
                    'null_count': int(df[col].isna().sum())
                }
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts()
                stats['categorical_columns'][col] = {
                    'unique_count': int(df[col].nunique()),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'null_count': int(df[col].isna().sum())
                }
        
        return stats
    
    async def _update_baselines(self, dataset_name: str, df: pd.DataFrame):
        """Update statistical baselines for future comparisons"""
        
        if dataset_name not in self.statistical_baselines:
            self.statistical_baselines[dataset_name] = {}
        
        baselines = self.statistical_baselines[dataset_name]
        
        # Update baselines for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in baselines:
                baselines[col] = {
                    'mean_history': [],
                    'std_history': [],
                    'min_history': [],
                    'max_history': []
                }
            
            # Add current statistics to history
            baselines[col]['mean_history'].append(float(df[col].mean()))
            baselines[col]['std_history'].append(float(df[col].std()))
            baselines[col]['min_history'].append(float(df[col].min()))
            baselines[col]['max_history'].append(float(df[col].max()))
            
            # Keep only last 100 measurements
            for key in baselines[col]:
                if len(baselines[col][key]) > 100:
                    baselines[col][key] = baselines[col][key][-100:]
    
    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """Determine quality level based on score"""
        if score >= 95:
            return DataQualityLevel.EXCELLENT
        elif score >= 85:
            return DataQualityLevel.GOOD
        elif score >= 70:
            return DataQualityLevel.ACCEPTABLE
        elif score >= 50:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNACCEPTABLE
    
    async def validate_streaming_data(self, record: Dict[str, Any], 
                                     dataset_name: str) -> bool:
        """Validate a single streaming record"""
        
        # Quick validation for streaming data
        if dataset_name not in self.validation_rules:
            return True
        
        rules = self.validation_rules[dataset_name]
        
        for rule in rules:
            if not rule.enabled or rule.field not in record:
                continue
            
            value = record[rule.field]
            
            try:
                if rule.rule_type == "range":
                    min_val = rule.parameters.get("min", float('-inf'))
                    max_val = rule.parameters.get("max", float('inf'))
                    
                    if value < min_val or value > max_val:
                        if rule.severity == "error":
                            return False
                
                elif rule.rule_type == "enum":
                    valid_values = rule.parameters.get("values", [])
                    if value not in valid_values:
                        if rule.severity == "error":
                            return False
                
            except Exception as e:
                logger.error(f"Error validating streaming record: {e}")
        
        return True
    
    async def generate_quality_report(self, dataset_name: str) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        if dataset_name not in self.validation_history:
            return {"error": "No validation history for dataset"}
        
        history = self.validation_history[dataset_name]
        
        if not history:
            return {"error": "Empty validation history"}
        
        # Get recent validations (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent = [v for v in history if v.timestamp > cutoff]
        
        if not recent:
            recent = history[-10:]  # Last 10 if no recent
        
        # Calculate aggregate metrics
        avg_quality_score = np.mean([v.quality_score for v in recent])
        min_quality_score = min(v.quality_score for v in recent)
        max_quality_score = max(v.quality_score for v in recent)
        
        # Count statuses
        status_counts = Counter(v.status.value for v in recent)
        
        # Aggregate anomalies
        all_anomalies = []
        for v in recent:
            all_anomalies.extend(v.anomalies)
        
        anomaly_types = Counter(a['type'] for a in all_anomalies)
        
        # Common issues
        all_warnings = []
        all_errors = []
        for v in recent:
            all_warnings.extend(v.warnings)
            all_errors.extend(v.errors)
        
        report = {
            'dataset': dataset_name,
            'report_timestamp': datetime.utcnow().isoformat(),
            'validation_count': len(recent),
            'time_range': {
                'start': min(v.timestamp for v in recent).isoformat(),
                'end': max(v.timestamp for v in recent).isoformat()
            },
            'quality_scores': {
                'average': round(avg_quality_score, 2),
                'min': round(min_quality_score, 2),
                'max': round(max_quality_score, 2),
                'current': round(recent[-1].quality_score, 2),
                'trend': 'improving' if recent[-1].quality_score > avg_quality_score else 'declining'
            },
            'quality_level': self._determine_quality_level(avg_quality_score).value,
            'validation_status': {
                'passed': status_counts.get('passed', 0),
                'failed': status_counts.get('failed', 0),
                'warning': status_counts.get('warning', 0)
            },
            'anomalies': {
                'total_count': len(all_anomalies),
                'by_type': dict(anomaly_types)
            },
            'common_issues': {
                'warnings': Counter(all_warnings).most_common(5),
                'errors': Counter(all_errors).most_common(5)
            },
            'latest_validation': {
                'timestamp': recent[-1].timestamp.isoformat(),
                'status': recent[-1].status.value,
                'quality_score': recent[-1].quality_score,
                'total_records': recent[-1].total_records,
                'invalid_records': recent[-1].invalid_records
            },
            'recommendations': self._generate_recommendations(recent)
        }
        
        return report
    
    def _generate_recommendations(self, validations: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation history"""
        recommendations = []
        
        # Check average quality score
        avg_score = np.mean([v.quality_score for v in validations])
        
        if avg_score < 70:
            recommendations.append(
                "Critical: Data quality is below acceptable threshold. Immediate action required."
            )
        elif avg_score < 85:
            recommendations.append(
                "Warning: Data quality needs improvement. Review validation rules and data sources."
            )
        
        # Check for recurring issues
        all_errors = []
        for v in validations:
            all_errors.extend(v.errors)
        
        if all_errors:
            error_counts = Counter(all_errors)
            most_common = error_counts.most_common(1)[0]
            recommendations.append(
                f"Address recurring error: {most_common[0]} (occurred {most_common[1]} times)"
            )
        
        # Check for anomaly trends
        anomaly_count = sum(len(v.anomalies) for v in validations)
        if anomaly_count > len(validations) * 5:  # More than 5 anomalies per validation
            recommendations.append(
                "High anomaly rate detected. Consider updating anomaly detection thresholds."
            )
        
        # Check for missing data trends
        missing_rates = []
        for v in validations:
            if 'missing_values' in v.field_stats:
                total_missing = sum(
                    stats['count'] 
                    for stats in v.field_stats['missing_values'].values()
                )
                missing_rate = (total_missing / (v.total_records * 10)) * 100  # Estimate
                missing_rates.append(missing_rate)
        
        if missing_rates and np.mean(missing_rates) > 5:
            recommendations.append(
                "High rate of missing values. Review data collection process."
            )
        
        if not recommendations:
            recommendations.append("Data quality is good. Continue monitoring.")
        
        return recommendations

async def main():
    """Example usage of data quality validator"""
    validator = DataQualityValidator()
    
    # Sample data for testing
    sample_transactions = [
        {
            'signature': 'tx1',
            'block_height': 1000,
            'block_timestamp': datetime.utcnow(),
            'net_profit': 50000,
            'profit_amount': 55000,
            'gas_cost': 5000,
            'roi_percentage': 10.0,
            'dex_count': 2
        },
        {
            'signature': 'tx2',
            'block_height': 1001,
            'block_timestamp': datetime.utcnow() - timedelta(hours=1),
            'net_profit': -1000,  # Loss
            'profit_amount': 4000,
            'gas_cost': 5000,
            'roi_percentage': -20.0,
            'dex_count': 3
        }
    ]
    
    # Validate dataset
    result = await validator.validate_dataset(
        'arbitrage_transactions',
        sample_transactions
    )
    
    print(f"Validation Status: {result.status.value}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Warnings: {result.warnings}")
    print(f"Errors: {result.errors}")
    
    # Generate quality report
    report = await validator.generate_quality_report('arbitrage_transactions')
    print(f"Quality Report: {json.dumps(report, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())