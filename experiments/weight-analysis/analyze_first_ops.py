#!/usr/bin/env python3
"""
Analyze the FIRST operation in each model to understand input preprocessing
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

def analyze_first_ops(model_path, label):
    print(f"\n{'='*70}")
    print(f"FIRST OPERATIONS: {label}")
    print(f"{'='*70}")

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    # Get ops
    try:
        ops = interp._get_ops_details()
    except:
        print("  Cannot get ops details")
        return

    print(f"\nFirst 10 operations:")
    for i, op in enumerate(ops[:10]):
        op_name = op.get('op_name', 'unknown')
        inputs = op.get('inputs', [])
        outputs = op.get('outputs', [])
        print(f"  {i}: {op_name}")
        print(f"     inputs: {inputs}")
        print(f"     outputs: {outputs}")

    # Get input tensor details
    inp = interp.get_input_details()[0]
    print(f"\nInput tensor:")
    print(f"  index: {inp['index']}")
    print(f"  name: {inp['name']}")
    print(f"  dtype: {inp['dtype']}")
    print(f"  shape: {inp['shape']}")
    qp = inp.get('quantization_parameters', {})
    if qp.get('scales') is not None and len(qp['scales']) > 0:
        print(f"  scale: {qp['scales'][0]}")
        print(f"  zero_point: {qp['zero_points'][0]}")
        s, zp = qp['scales'][0], qp['zero_points'][0]
        print(f"  uint8 0 → float {(0 - zp) * s:.4f}")
        print(f"  uint8 127 → float {(127 - zp) * s:.4f}")
        print(f"  uint8 255 → float {(255 - zp) * s:.4f}")

models = [
    (MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite", "ST per-tensor"),
    (MODELS_DIR / "quantized-pt.tflite", "My per-tensor"),
    (MODELS_DIR / "quantized-pc.tflite", "My per-channel"),
]

for path, label in models:
    if path.exists():
        analyze_first_ops(str(path), label)
