#!/usr/bin/env python3
"""
Test script to verify the modified cross-adaptation functionality
with KLIEP and KMM adaptation methods.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from adapt.instance_based import KLIEP, KMM
from src.cross_adaptation.core.cross_adaptation import Adapter


def create_synthetic_data():
    """Create synthetic datasets for testing"""
    np.random.seed(42)
    
    train_data = {}
    test_data = {}
    
    # Create 3 synthetic domains with different distributions
    for i, domain in enumerate(['Domain_A', 'Domain_B', 'Domain_C']):
        n_train = 150
        n_test = 50
        n_features = 5
        
        # Create features with domain-specific shift
        X_train = np.random.randn(n_train, n_features) + i * 0.3
        X_test = np.random.randn(n_test, n_features) + i * 0.3
        
        # Create targets with some noise
        y_train = (X_train[:, 0] + X_train[:, 1] > i * 0.2).astype(int)
        y_test = (X_test[:, 0] + X_test[:, 1] > i * 0.2).astype(int)
        
        # Add some label noise
        noise_idx_train = np.random.choice(n_train, size=int(0.1 * n_train), replace=False)
        y_train[noise_idx_train] = 1 - y_train[noise_idx_train]
        
        # Create DataFrames
        train_df = pd.DataFrame(X_train, columns=[f'feature_{j}' for j in range(n_features)])
        train_df['target'] = y_train
        
        test_df = pd.DataFrame(X_test, columns=[f'feature_{j}' for j in range(n_features)])
        test_df['target'] = y_test
        
        train_data[domain] = train_df
        test_data[domain] = test_df
    
    return train_data, test_data


def test_kliep_adaptation():
    """Test cross-adaptation with KLIEP"""
    print("\n" + "="*60)
    print("Testing Cross-Adaptation with KLIEP")
    print("="*60)
    
    # Create synthetic data
    train_data, test_data = create_synthetic_data()
    
    # Initialize models
    adapt_model = KLIEP(
        kernel='rbf',
        max_centers=100,
        lr=0.01,
        max_iter=500
    )
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Create adapter
    adapter = Adapter(
        train_data=train_data,
        adapt_model=adapt_model,
        estimator=estimator
    )
    
    # Adapt the data and save
    save_dir = "test_adapted_data_kliep"
    adapted_data = adapter.adapt(save_dir=save_dir)
    
    print(f"\n✓ Data adapted and saved to {save_dir}")
    for domain, data in adapted_data.items():
        print(f"  - {domain}: {len(data['X'])} samples, weights range: [{data['weights'].min():.3f}, {data['weights'].max():.3f}]")
    
    # Train on adapted data
    metrics = [accuracy_score, f1_score]
    results = adapter.train_on_adapted_data(
        test_data=test_data,
        metrics=metrics,
        use_weights=True,
        save_model=True
    )
    
    print("\n✓ Model trained on adapted data")
    print("Results on test data:")
    for key, value in results.items():
        print(f"  - {key}: {value:.4f}")
    
    # Calculate baseline
    baseline_results = adapter.calc_baseline(test_data, metrics)
    
    print("\nBaseline results (no adaptation):")
    for key, value in baseline_results.items():
        print(f"  - {key}: {value:.4f}")
    
    # Compare results
    compare_results = adapter.compare(results, baseline_results, metrics, test_data)
    
    print("\nImprovement over baseline:")
    for key, value in compare_results.items():
        print(f"  - {key}: {value:+.4f}")
    
    return results, baseline_results


def test_kmm_adaptation():
    """Test cross-adaptation with KMM"""
    print("\n" + "="*60)
    print("Testing Cross-Adaptation with KMM")
    print("="*60)
    
    # Create synthetic data
    train_data, test_data = create_synthetic_data()
    
    # Initialize models
    adapt_model = KMM(
        kernel='rbf',
        B=10,
        eps=1e-3,
        max_iter=100
    )
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Create adapter
    adapter = Adapter(
        train_data=train_data,
        adapt_model=adapt_model,
        estimator=estimator
    )
    
    # Adapt the data and save
    save_dir = "test_adapted_data_kmm"
    adapted_data = adapter.adapt(save_dir=save_dir)
    
    print(f"\n✓ Data adapted and saved to {save_dir}")
    for domain, data in adapted_data.items():
        print(f"  - {domain}: {len(data['X'])} samples, weights range: [{data['weights'].min():.3f}, {data['weights'].max():.3f}]")
    
    # Train on adapted data
    metrics = [accuracy_score, f1_score]
    results = adapter.train_on_adapted_data(
        test_data=test_data,
        metrics=metrics,
        use_weights=True,
        save_model=True
    )
    
    print("\n✓ Model trained on adapted data")
    print("Results on test data:")
    for key, value in results.items():
        print(f"  - {key}: {value:.4f}")
    
    # Calculate baseline
    baseline_results = adapter.calc_baseline(test_data, metrics)
    
    print("\nBaseline results (no adaptation):")
    for key, value in baseline_results.items():
        print(f"  - {key}: {value:.4f}")
    
    # Compare results
    compare_results = adapter.compare(results, baseline_results, metrics, test_data)
    
    print("\nImprovement over baseline:")
    for key, value in compare_results.items():
        print(f"  - {key}: {value:+.4f}")
    
    return results, baseline_results


def test_load_adapted_data():
    """Test loading previously saved adapted data"""
    print("\n" + "="*60)
    print("Testing Load Adapted Data Functionality")
    print("="*60)
    
    # Create synthetic data for testing
    train_data, test_data = create_synthetic_data()
    
    # Initialize models
    adapt_model = KLIEP(kernel='rbf')
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Create adapter
    adapter = Adapter(
        train_data=train_data,
        adapt_model=adapt_model,
        estimator=estimator
    )
    
    # Try to load previously saved data
    if os.path.exists("test_adapted_data_kliep"):
        loaded_data = adapter.load_adapted_data("test_adapted_data_kliep")
        print(f"\n✓ Successfully loaded adapted data from test_adapted_data_kliep")
        for domain, data in loaded_data.items():
            print(f"  - {domain}: {len(data['X'])} samples loaded")
        
        # Train on loaded data
        metrics = [accuracy_score, f1_score]
        results = adapter.train_on_adapted_data(
            test_data=test_data,
            metrics=metrics,
            use_weights=True,
            save_model=False
        )
        
        print("\n✓ Model trained on loaded adapted data")
        print("Results on test data:")
        for key, value in results.items():
            print(f"  - {key}: {value:.4f}")
    else:
        print("\n⚠ No previously saved data found. Run test_kliep_adaptation() first.")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CROSS-ADAPTATION MODULE TEST SUITE")
    print("="*60)
    print("\nThis script tests the modified cross-adaptation module with:")
    print("  1. KLIEP adaptation method")
    print("  2. KMM adaptation method")
    print("  3. Data saving and loading functionality")
    print("  4. Training on adapted data with instance weights")
    
    try:
        # Test KLIEP
        kliep_results, kliep_baseline = test_kliep_adaptation()
        
        # Test KMM
        kmm_results, kmm_baseline = test_kmm_adaptation()
        
        # Test loading functionality
        test_load_adapted_data()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        
        # Clean up test directories (optional)
        print("\nTest directories created:")
        print("  - test_adapted_data_kliep/")
        print("  - test_adapted_data_kmm/")
        print("  - adapted_estimator.bin")
        print("  - baseline_estimator.bin")
        print("  - scaler.bin")
        print("\nYou can delete these files if no longer needed.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
