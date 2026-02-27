#!/usr/bin/env python3
"""
Deep dive into ST model - look for any hints about how it was created
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ST_MODEL = MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite"
MY_MODEL = MODELS_DIR / "quantized-pt.tflite"

def analyze_model_deep(path, label):
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS: {label}")
    print(f"{'='*70}")

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()

    # Get all tensor details
    tensors = interp.get_tensor_details()

    # Look at first layer (input conv)
    print("\n--- First few tensor names ---")
    for t in tensors[:10]:
        print(f"  {t['index']:3d}: {t['name'][:70]}")
        print(f"       shape={t['shape']} dtype={t['dtype']}")

    # Look for any metadata or version info in names
    print("\n--- Looking for version/source hints ---")
    for t in tensors:
        name = t['name'].lower()
        if any(kw in name for kw in ['version', 'tf', 'keras', 'mobilenet', 'model']):
            print(f"  {t['name'][:80]}")

    # Check quantization consistency across layers
    print("\n--- Quantization parameter patterns ---")
    scale_samples = []
    zp_samples = []
    for t in tensors:
        qp = t.get('quantization_parameters', {})
        scales = qp.get('scales', np.array([]))
        zps = qp.get('zero_points', np.array([]))
        if len(scales) == 1:
            scale_samples.append(scales[0])
            zp_samples.append(zps[0])

    if scale_samples:
        scales = np.array(scale_samples)
        zps = np.array(zp_samples)
        print(f"  Total per-tensor quantized: {len(scale_samples)}")
        print(f"  Scale range: [{scales.min():.6f}, {scales.max():.6f}]")
        print(f"  Zero-point range: [{zps.min()}, {zps.max()}]")
        print(f"  Zero-points == 0: {np.sum(zps == 0)}")
        print(f"  Zero-points == 128: {np.sum(zps == 128)}")
        print(f"  Zero-points == 127: {np.sum(zps == 127)}")

        # Distribution of zero points
        unique_zps, counts = np.unique(zps, return_counts=True)
        print(f"  Zero-point distribution (top 5):")
        sorted_idx = np.argsort(counts)[::-1][:5]
        for i in sorted_idx:
            print(f"    zp={unique_zps[i]}: {counts[i]} tensors")

def main():
    analyze_model_deep(ST_MODEL, "ST per-tensor (WORKS)")
    analyze_model_deep(MY_MODEL, "My per-tensor (BROKEN)")

if __name__ == "__main__":
    main()
