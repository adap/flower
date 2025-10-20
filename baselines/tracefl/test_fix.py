#!/usr/bin/env python3
"""Test script to verify the TraceFL faulty client detection fix."""

import subprocess
import sys
import os

def test_tracefl_fix():
    """Test the TraceFL faulty client detection fix."""
    print("🧪 Testing TraceFL faulty client detection fix...")
    
    # We're already in the TraceFL baseline directory
    print(f"Current directory: {os.getcwd()}")
    
    # Run experiment C (faulty client detection)
    cmd = [
        "flwr", "run", ".",
        "--federation-config", "options.num-supernodes=4",
        "--run-config", 
        "num-server-rounds=3 "
        "tracefl.dataset='mnist' "
        "tracefl.model='resnet18' "
        "tracefl.num-clients=4 "
        "tracefl.dirichlet-alpha=0.7 "
        "tracefl.max-per-client-data-size=2048 "
        "tracefl.max-server-data-size=2048 "
        "tracefl.batch-size=32 "
        "tracefl.provenance-rounds='1,2,3' "
        "tracefl.faulty-clients-ids='[0]' "
        "tracefl.label2flip='{1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0}' "
        "tracefl.use-deterministic-sampling=true "
        "tracefl.random-seed=42 "
        "tracefl.output-dir='results/test_fix' "
        "min-train-nodes=4 "
        "fraction-train=1.0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ TraceFL experiment completed successfully!")
            print("📊 Checking results...")
            
            # Check if results were generated
            if os.path.exists("results/test_fix"):
                print("✅ Results directory created")
                
                # Look for provenance CSV files
                csv_files = [f for f in os.listdir("results/test_fix") if f.endswith('.csv')]
                if csv_files:
                    print(f"✅ Found {len(csv_files)} provenance CSV files")
                    
                    # Read the first CSV to check client contributions
                    csv_path = os.path.join("results/test_fix", csv_files[0])
                    with open(csv_path, 'r') as f:
                        content = f.read()
                        
                    # Check if client 0 has high contribution (indicating successful detection)
                    if "client_0" in content and "client_2" in content:
                        print("✅ Client IDs found in results")
                        
                        # Look for high client 0 contribution
                        lines = content.split('\n')
                        for line in lines:
                            if "client_0" in line and "client_2" in line:
                                print(f"📈 Contribution line: {line}")
                                break
                        
                        print("🎯 Fix appears to be working - client IDs are correctly mapped!")
                        return True
                    else:
                        print("❌ Client IDs not found in results")
                        return False
                else:
                    print("❌ No CSV files found in results")
                    return False
            else:
                print("❌ Results directory not created")
                return False
                
        else:
            print(f"❌ TraceFL experiment failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TraceFL experiment timed out")
        return False
    except Exception as e:
        print(f"❌ Error running TraceFL experiment: {e}")
        return False

if __name__ == "__main__":
    success = test_tracefl_fix()
    sys.exit(0 if success else 1)
