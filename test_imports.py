"""
Test imports to diagnose which module is hanging
"""
import sys
import time

print("Testing imports...")

try:
    print("1. Testing streamlit...", end=" ", flush=True)
    import streamlit as st
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("2. Testing pandas...", end=" ", flush=True)
    import pandas as pd
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("3. Testing numpy...", end=" ", flush=True)
    import numpy as np
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("4. Testing sklearn...", end=" ", flush=True)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("5. Testing tensorflow (this may take time)...", end=" ", flush=True)
    import tensorflow as tf
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("6. Testing data_preprocessing...", end=" ", flush=True)
    from data_preprocessing import DataPreprocessor
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("7. Testing optimization...", end=" ", flush=True)
    from optimization import HybridGAPSO
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("8. Testing transformer_model...", end=" ", flush=True)
    from transformer_model import GeneTransformerClassifier
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("9. Testing evaluation...", end=" ", flush=True)
    from evaluation import ModelEvaluator
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("10. Testing interpretability...", end=" ", flush=True)
    from interpretability import GeneImportanceAnalyzer
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
