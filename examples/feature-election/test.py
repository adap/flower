"""
Feature Election - Comprehensive Test Suite

Combines basic verification (imports, datasets) with advanced feature testing
(auto-tuning, hill climbing, communication metrics, and phase logic).
"""

import logging
import sys

import numpy as np

# Configure logging to suppress internal strategy logs during testing
# so the test output remains clean and readable.
logging.basicConfig(level=logging.CRITICAL)


def main():
    print("=" * 80)
    print(" Feature Election - Tests")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------------
    # PART 1: Basic Environment & Logic Verification
    # ------------------------------------------------------------------------

    print("--- Part 1: Basic Verification ---")

    # Test 1: Import check
    print("1. Testing core imports...")
    try:
        from feature_election.feature_election_utils import FeatureSelector
        from feature_election.strategy import FeatureElectionStrategy
        from feature_election.task import create_synthetic_dataset
        from flwr.app import (
            ArrayRecord,
            ConfigRecord,
        )
        from flwr.common.record import Array

        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        print("   → Install dependencies: pip install -e .")
        return 1

    # Test 2: Create dataset
    print("\n2. Creating synthetic dataset...")
    try:
        df, feature_names = create_synthetic_dataset(
            n_samples=100, n_features=35, n_informative=15, n_redundant=10, n_repeated=10
        )
        print(f"   ✓ Dataset created: {df.shape[0]} samples, {len(feature_names)} features")
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {e}")
        return 1

    # Test 3: Feature selection
    print("\n3. Testing basic feature selection (Lasso)...")
    try:
        X = df.drop(columns=["target"]).values
        y = df["target"].values

        selector = FeatureSelector(fs_method="lasso")
        mask, scores = selector.select_features(X, y)

        print("   ✓ Feature selection successful")
        print(f"   Selected {np.sum(mask)}/{len(mask)} features")
    except Exception as e:
        print(f"   ✗ Feature selection failed: {e}")
        return 1

    # Test 4: Strategy initialization
    print("\n4. Testing Strategy initialization...")
    try:
        strategy = FeatureElectionStrategy(
            freedom_degree=0.5,
            aggregation_mode="weighted",
        )
        print("   ✓ Strategy initialized successfully")
    except Exception as e:
        print(f"   ✗ Strategy initialization failed: {e}")
        return 1

    # Test 5: Basic Aggregation logic
    print("\n5. Testing Basic Aggregation logic...")
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

        print("   ✓ Aggregation successful")
        print(f"   Global selection: {np.sum(global_mask)}/{len(global_mask)} features")
    except Exception as e:
        print(f"   ✗ Aggregation failed: {e}")
        return 1

    # Test 6: ArrayRecord functionality
    print("\n6. Testing ArrayRecord functionality...")
    try:
        arr = ArrayRecord()
        arr["test"] = Array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        retrieved = arr["test"].numpy()
        assert len(retrieved) == 3
        print("   ✓ ArrayRecord works correctly")
    except Exception as e:
        print(f"   ✗ ArrayRecord test failed: {e}")
        return 1

    # ------------------------------------------------------------------------
    # PART 2: Advanced Features Verification (Hill Climbing & Metrics)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("--- Part 2: Advanced Features Verification ---")

    # Test 7: Hill Climbing Logic
    print("7. Testing Hill Climbing (Auto-Tuning) Logic...")
    try:
        strategy = FeatureElectionStrategy(freedom_degree=0.5, tuning_rounds=3, auto_tune=True)

        # --- Scenario A: Improvement -> Continue ---
        # History: Round T-1 (0.5 -> 0.80 acc), Round T (0.6 -> 0.82 acc)
        strategy.tuning_history = [(0.5, 0.80), (0.6, 0.82)]
        strategy.current_direction = 1
        strategy.search_step = 0.1

        next_fd = strategy._calculate_next_fd()
        expected_fd = 0.7

        if abs(next_fd - expected_fd) < 1e-6:
            print(f"   ✓ Scenario A (Improvement): Correctly moved to {next_fd:.2f}")
        else:
            print(f"   ✗ Scenario A Failed: Expected {expected_fd}, got {next_fd}")
            return 1

        # --- Scenario B: Degradation -> Reverse & Decay ---
        # History: Round T-1 (0.5 -> 0.80 acc), Round T (0.6 -> 0.75 acc)
        strategy.tuning_history = [(0.5, 0.80), (0.6, 0.75)]
        strategy.current_direction = 1
        strategy.search_step = 0.1

        next_fd = strategy._calculate_next_fd()
        # Calculation: 0.5 + (-1 * 0.05) = 0.45 (Step decayed to 0.05)
        expected_fd = 0.45

        if abs(next_fd - expected_fd) < 1e-6:
            print(f"   ✓ Scenario B (Degradation): Correctly reversed/decayed to {next_fd:.2f}")
        else:
            print(f"   ✗ Scenario B Failed: Expected {expected_fd}, got {next_fd}")
            return 1

        # --- Scenario C: Boundary Checks ---
        strategy.tuning_history = [(0.1, 0.80), (0.05, 0.82)]  # Heading down
        strategy.current_direction = -1
        strategy.search_step = 0.1  # Big step

        # Next would be 0.05 + (-1 * 0.1) = -0.05. Should clip to MIN_FD (0.05)
        next_fd = strategy._calculate_next_fd()
        if next_fd == 0.05:
            print(f"   ✓ Scenario C (Clipping): Correctly clipped to minimum {next_fd}")
        else:
            print(f"   ✗ Scenario C Failed: Expected 0.05, got {next_fd}")
            return 1

    except Exception as e:
        print(f"   ✗ Hill Climbing test crashed: {e}")
        return 1

    # Test 8: Byte Counting
    print("\n8. Testing Communication Cost Tracking...")
    try:
        # Create a dummy payload: 100 float32 elements = 400 bytes
        data = np.zeros(100, dtype=np.float32)
        arrays = ArrayRecord({"test_data": Array(data)})

        # Access internal helper (strictly for testing)
        calculated_size = strategy._calculate_payload_size(arrays)

        if calculated_size == 400:
            print(f"   ✓ Byte counting accurate: {calculated_size} bytes")
        else:
            print(f"   ✗ Byte counting failed: Expected 400, got {calculated_size}")
            return 1

    except Exception as e:
        print(f"   ✗ Metric test failed: {e}")
        return 1

    # Test 9: Phase Switching Logic
    print("\n9. Testing Phase Switching Logic...")
    try:
        # Init strategy: 1 Round Selection + 2 Rounds Tuning = 3 Rounds Setup.
        strategy = FeatureElectionStrategy(tuning_rounds=2, auto_tune=True)

        # Mock Grid object
        class MockGrid:
            def get_node_ids(self):
                return [1, 2, 3]

        grid = MockGrid()

        # --- Round 1: Selection ---
        msgs = strategy.configure_train(
            server_round=1, arrays=ArrayRecord(), config=ConfigRecord(), grid=grid
        )
        phase_r1 = msgs[0].content["config"]["phase"]
        if phase_r1 == "feature_selection":
            print("   ✓ Round 1: Correctly set to 'feature_selection'")
        else:
            print(f"   ✗ Round 1: Expected 'feature_selection', got '{phase_r1}'")
            return 1

        # --- Mock Cached Selections (Required for Tuning Phase) ---
        strategy.cached_client_selections = {
            "1": {
                "selected_features": np.array([True, False], dtype=bool),
                "feature_scores": np.array([1.0, 0.0]),
                "num_samples": 10,
            }
        }

        # --- Round 2: Tuning ---
        msgs = strategy.configure_train(
            server_round=2, arrays=ArrayRecord(), config=ConfigRecord(), grid=grid
        )
        phase_r2 = msgs[0].content["config"]["phase"]
        if phase_r2 == "tuning_eval":
            print("   ✓ Round 2: Correctly set to 'tuning_eval'")
        else:
            print(f"   ✗ Round 2: Expected 'tuning_eval', got '{phase_r2}'")
            return 1

        # --- Round 3: Tuning ---
        msgs = strategy.configure_train(
            server_round=3, arrays=ArrayRecord(), config=ConfigRecord(), grid=grid
        )
        phase_r3 = msgs[0].content["config"]["phase"]
        if phase_r3 == "tuning_eval":
            print("   ✓ Round 3: Correctly set to 'tuning_eval'")
        else:
            print(f"   ✗ Round 3: Expected 'tuning_eval', got '{phase_r3}'")
            return 1

        # --- Round 4: FL Training ---
        msgs = strategy.configure_train(
            server_round=4, arrays=ArrayRecord(), config=ConfigRecord(), grid=grid
        )
        phase_r4 = msgs[0].content["config"]["phase"]
        if phase_r4 == "fl_training":
            print("   ✓ Round 4: Correctly set to 'fl_training'")
        else:
            print(f"   ✗ Round 4: Expected 'fl_training', got '{phase_r4}'")
            return 1

    except Exception as e:
        print(f"   ✗ Phase logic test failed: {e}")
        return 1

    # Success!
    print("\n" + "=" * 80)
    print(" ✓ All 9 tests passed!")
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
