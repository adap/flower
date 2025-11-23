"""
Quick Test Script for Feature Election

Run this to quickly verify the installation works correctly.
"""

import sys
from strategy import FeatureElectionStrategy
from client_app import FeatureElectionClient
from feature_election_utils import FeatureSelector
from task import create_synthetic_dataset

def main():
    """Run quick verification tests."""
    
    print("=" * 70)
    print(" Feature Election - Quick Verification")
    print("=" * 70)
    print()
    
    # Test 1: Import check
    print("1. Testing imports...")
    try:
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        print("   → Install dependencies: pip install -e .")
        return 1
    
    # Test 2: Create dataset
    print("\n2. Creating synthetic dataset...")
    try:
        df, feature_names = create_synthetic_dataset(
            n_samples=100,
            n_features=35,
            n_informative=15,
            n_redundant=10,
            n_repeated=10
        )
        print(f"   ✓ Dataset created: {df.shape[0]} samples, {len(feature_names)} features")
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {e}")
        return 1
    
    # Test 3: Feature selection
    print("\n3. Testing feature selection...")
    try:
        import numpy as np
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        
        selector = FeatureSelector(fs_method="lasso")
        mask, scores = selector.select_features(X, y)
        
        print(f"   ✓ Feature selection successful")
        print(f"   Selected {np.sum(mask)}/{len(mask)} features")
    except Exception as e:
        print(f"   ✗ Feature selection failed: {e}")
        return 1
    
    # Test 4: Strategy initialization
    print("\n4. Testing strategy...")
    try:
        strategy = FeatureElectionStrategy(
            freedom_degree=0.5,
            aggregation_mode="weighted",
        )
        print("   ✓ Strategy initialized successfully")
    except Exception as e:
        print(f"   ✗ Strategy initialization failed: {e}")
        return 1
    
    # Test 5: Aggregation
    print("\n5. Testing aggregation...")
    try:
        # Create dummy client selections
        client_selections = {
            "0": {
                "selected_features": np.random.rand(20) > 0.5,
                "feature_scores": np.random.rand(20),
                "num_samples": 50,
                "initial_score": 0.7,
                "fs_score": 0.75,
            },
            "1": {
                "selected_features": np.random.rand(20) > 0.5,
                "feature_scores": np.random.rand(20),
                "num_samples": 50,
                "initial_score": 0.72,
                "fs_score": 0.77,
            },
        }
        
        strategy.num_features = 20
        global_mask = strategy._aggregate_selections(client_selections)
        
        print(f"   ✓ Aggregation successful")
        print(f"   Global selection: {np.sum(global_mask)}/{len(global_mask)} features")
    except Exception as e:
        print(f"   ✗ Aggregation failed: {e}")
        return 1
    
    # Success!
    print("\n" + "=" * 70)
    print(" ✓ All tests passed!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run full simulation: flwr run .")
    print("  2. Try examples: python examples/basic_usage.py")
    print("  3. Run test suite: pytest tests/ -v")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
