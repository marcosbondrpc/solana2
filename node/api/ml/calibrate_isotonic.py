#!/usr/bin/env python3
"""
Isotonic Calibration for MEV Profit Predictions
Ensures monotonic probability mapping for better decision making
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from typing import List, Tuple, Dict, Optional
import json
import struct
import mmap
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CalibrationResult:
    """Result of isotonic calibration"""
    original_scores: np.ndarray
    calibrated_scores: np.ndarray
    calibration_map: List[Tuple[float, float]]
    metrics: Dict[str, float]


class IsotonicCalibrator:
    """
    High-performance isotonic calibrator for MEV profit predictions
    Exports to binary LUT for ultra-fast lookups in production
    """
    
    def __init__(self, n_bins: int = 1000):
        self.n_bins = n_bins
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.calibration_map = None
        self.lut = None
        self.min_score = 0.0
        self.max_score = 1.0
        
    def fit(self, 
            predicted_profits: np.ndarray, 
            actual_profits: np.ndarray,
            sample_weights: Optional[np.ndarray] = None) -> CalibrationResult:
        """
        Fit isotonic regression for profit calibration
        
        Args:
            predicted_profits: Model predictions (0-1 normalized)
            actual_profits: Actual profits (0-1 normalized)
            sample_weights: Optional weights for samples
        """
        # Ensure monotonicity by sorting
        sorted_indices = np.argsort(predicted_profits)
        X = predicted_profits[sorted_indices]
        y = actual_profits[sorted_indices]
        
        if sample_weights is not None:
            weights = sample_weights[sorted_indices]
        else:
            weights = None
            
        # Fit isotonic regression
        self.isotonic.fit(X, y, sample_weight=weights)
        
        # Generate calibration map
        self.min_score = X.min()
        self.max_score = X.max()
        
        # Create fine-grained calibration points
        calibration_points = np.linspace(self.min_score, self.max_score, self.n_bins)
        calibrated_values = self.isotonic.transform(calibration_points)
        
        self.calibration_map = list(zip(calibration_points, calibrated_values))
        
        # Build LUT for fast lookups
        self._build_lut()
        
        # Calculate metrics
        calibrated_predictions = self.isotonic.transform(predicted_profits)
        metrics = self._calculate_metrics(actual_profits, predicted_profits, calibrated_predictions)
        
        return CalibrationResult(
            original_scores=predicted_profits,
            calibrated_scores=calibrated_predictions,
            calibration_map=self.calibration_map,
            metrics=metrics
        )
    
    def _build_lut(self):
        """Build binary lookup table for O(1) calibration"""
        self.lut = np.zeros(self.n_bins, dtype=np.float32)
        
        for i, (score, calibrated) in enumerate(self.calibration_map):
            self.lut[i] = calibrated
            
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new scores using fast LUT
        
        Args:
            scores: Uncalibrated scores
            
        Returns:
            Calibrated scores
        """
        if self.lut is None:
            raise ValueError("Calibrator not fitted yet")
            
        # Clip to valid range
        scores = np.clip(scores, self.min_score, self.max_score)
        
        # Fast LUT lookup with linear interpolation
        indices = (scores - self.min_score) / (self.max_score - self.min_score) * (self.n_bins - 1)
        lower_indices = np.floor(indices).astype(int)
        upper_indices = np.ceil(indices).astype(int)
        
        # Interpolation weights
        weights = indices - lower_indices
        
        # Lookup and interpolate
        lower_values = self.lut[lower_indices]
        upper_values = self.lut[upper_indices]
        
        return lower_values * (1 - weights) + upper_values * weights
    
    def export_lut(self, filepath: str):
        """
        Export LUT to binary file for Rust/C++ integration
        
        Format:
        - Header: magic(4), version(2), n_bins(4), min_score(4), max_score(4)
        - Data: float32 array of calibrated values
        """
        if self.lut is None:
            raise ValueError("Calibrator not fitted yet")
            
        with open(filepath, 'wb') as f:
            # Write header
            f.write(b'ISOT')  # Magic number
            f.write(struct.pack('H', 1))  # Version
            f.write(struct.pack('I', self.n_bins))
            f.write(struct.pack('f', self.min_score))
            f.write(struct.pack('f', self.max_score))
            
            # Write LUT data
            self.lut.tofile(f)
            
        print(f"Exported LUT to {filepath} ({os.path.getsize(filepath)} bytes)")
    
    def export_mmap_lut(self, filepath: str):
        """
        Export memory-mapped LUT for zero-copy access
        """
        if self.lut is None:
            raise ValueError("Calibrator not fitted yet")
            
        # Calculate total size
        header_size = 18  # 4 + 2 + 4 + 4 + 4
        data_size = self.n_bins * 4  # float32
        total_size = header_size + data_size
        
        # Create memory-mapped file
        with open(filepath, 'wb') as f:
            f.write(b'\0' * total_size)
            
        with open(filepath, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), total_size)
            
            # Write header
            mm[0:4] = b'ISOT'
            mm[4:6] = struct.pack('H', 1)
            mm[6:10] = struct.pack('I', self.n_bins)
            mm[10:14] = struct.pack('f', self.min_score)
            mm[14:18] = struct.pack('f', self.max_score)
            
            # Write LUT data
            mm[18:] = self.lut.tobytes()
            
            mm.close()
            
        print(f"Exported memory-mapped LUT to {filepath}")
    
    def _calculate_metrics(self, 
                          actual: np.ndarray, 
                          predicted: np.ndarray, 
                          calibrated: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics"""
        
        # Expected Calibration Error (ECE)
        ece_original = self._calculate_ece(actual, predicted)
        ece_calibrated = self._calculate_ece(actual, calibrated)
        
        # Maximum Calibration Error (MCE)
        mce_original = self._calculate_mce(actual, predicted)
        mce_calibrated = self._calculate_mce(actual, calibrated)
        
        # Brier Score
        brier_original = np.mean((predicted - actual) ** 2)
        brier_calibrated = np.mean((calibrated - actual) ** 2)
        
        return {
            'ece_original': ece_original,
            'ece_calibrated': ece_calibrated,
            'ece_improvement': ece_original - ece_calibrated,
            'mce_original': mce_original,
            'mce_calibrated': mce_calibrated,
            'mce_improvement': mce_original - mce_calibrated,
            'brier_original': brier_original,
            'brier_calibrated': brier_calibrated,
            'brier_improvement': brier_original - brier_calibrated,
        }
    
    def _calculate_ece(self, actual: np.ndarray, predicted: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (predicted >= bin_boundaries[i]) & (predicted < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = actual[mask].mean()
                bin_confidence = predicted[mask].mean()
                bin_weight = mask.sum() / len(predicted)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
                
        return ece
    
    def _calculate_mce(self, actual: np.ndarray, predicted: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            mask = (predicted >= bin_boundaries[i]) & (predicted < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = actual[mask].mean()
                bin_confidence = predicted[mask].mean()
                mce = max(mce, abs(bin_accuracy - bin_confidence))
                
        return mce


class AdaptiveCalibrator:
    """
    Adaptive calibrator that updates with new data streams
    """
    
    def __init__(self, window_size: int = 10000, update_frequency: int = 100):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.calibrator = IsotonicCalibrator()
        self.buffer_X = []
        self.buffer_y = []
        self.update_count = 0
        
    def update(self, predicted: float, actual: float):
        """Update calibrator with new observation"""
        self.buffer_X.append(predicted)
        self.buffer_y.append(actual)
        
        # Maintain window size
        if len(self.buffer_X) > self.window_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)
            
        self.update_count += 1
        
        # Refit periodically
        if self.update_count % self.update_frequency == 0 and len(self.buffer_X) >= 100:
            self.refit()
            
    def refit(self):
        """Refit calibrator with buffered data"""
        X = np.array(self.buffer_X)
        y = np.array(self.buffer_y)
        self.calibrator.fit(X, y)
        
    def calibrate(self, score: float) -> float:
        """Apply calibration to single score"""
        if self.calibrator.lut is None:
            return score  # No calibration available yet
        return self.calibrator.calibrate(np.array([score]))[0]


def generate_test_data(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test data for calibration"""
    np.random.seed(42)
    
    # Generate predicted profits with systematic bias
    predicted = np.random.beta(2, 5, n_samples)
    
    # Generate actual profits with noise and bias
    noise = np.random.normal(0, 0.1, n_samples)
    actual = np.clip(predicted ** 1.5 + noise, 0, 1)  # Systematic underestimation
    
    return predicted, actual


def main():
    """Demonstration of isotonic calibration"""
    
    # Generate test data
    print("Generating test data...")
    predicted, actual = generate_test_data(10000)
    
    # Split into train/test
    split_idx = int(len(predicted) * 0.8)
    train_pred, test_pred = predicted[:split_idx], predicted[split_idx:]
    train_actual, test_actual = actual[:split_idx], actual[split_idx:]
    
    # Create and fit calibrator
    print("\nFitting isotonic calibrator...")
    calibrator = IsotonicCalibrator(n_bins=1000)
    result = calibrator.fit(train_pred, train_actual)
    
    # Print metrics
    print("\nCalibration Metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Test calibration on test set
    print("\nTesting calibration on test set...")
    test_calibrated = calibrator.calibrate(test_pred)
    
    # Calculate test metrics
    test_ece_original = calibrator._calculate_ece(test_actual, test_pred)
    test_ece_calibrated = calibrator._calculate_ece(test_actual, test_calibrated)
    
    print(f"Test ECE Original: {test_ece_original:.6f}")
    print(f"Test ECE Calibrated: {test_ece_calibrated:.6f}")
    print(f"Test ECE Improvement: {test_ece_original - test_ece_calibrated:.6f}")
    
    # Export LUTs
    print("\nExporting LUTs...")
    calibrator.export_lut("calibration_lut.bin")
    calibrator.export_mmap_lut("calibration_lut.mmap")
    
    # Test adaptive calibrator
    print("\nTesting adaptive calibrator...")
    adaptive = AdaptiveCalibrator(window_size=1000, update_frequency=50)
    
    for i in range(1000):
        adaptive.update(predicted[i], actual[i])
        
    # Test single prediction
    test_score = 0.7
    calibrated_score = adaptive.calibrate(test_score)
    print(f"Single score calibration: {test_score:.3f} -> {calibrated_score:.3f}")
    
    print("\nCalibration complete!")


if __name__ == "__main__":
    main()