#!/usr/bin/env python3
"""
Deep structural analysis: Compare ST model structure vs our model structure
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import Counter

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ST_MODEL = MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite"
MY_MODEL = MODELS_DIR / "quantized-pt.tflite"

def analyze_model_ops(path, label):
    """Analyze the operations in a model"""
    print(f"\n{'='*70}")
    print(f"MODEL STRUCTURE: {label}")
    print(f"Path: {path}")
    print(f"{'='*70}")

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()

    # Get all tensor details
    tensors = interp.get_tensor_details()

    # Categorize tensors by shape patterns
    shape_counts = Counter()
    tensor_types = {'weights_4d': [], 'bias_1d': [], 'other': []}

    for t in tensors:
        shape = tuple(t['shape'])
        shape_counts[len(shape)] += 1

        if len(shape) == 4:
            tensor_types['weights_4d'].append({
                'name': t['name'],
                'shape': shape,
                'scales': t.get('quantization_parameters', {}).get('scales', np.array([]))
            })
        elif len(shape) == 1:
            tensor_types['bias_1d'].append({
                'name': t['name'],
                'shape': shape
            })

    print(f"\nTotal tensors: {len(tensors)}")
    print(f"Tensor dimension distribution: {dict(shape_counts)}")
    print(f"4D weight tensors: {len(tensor_types['weights_4d'])}")
    print(f"1D bias tensors: {len(tensor_types['bias_1d'])}")

    # Analyze weight tensor shapes
    weight_shapes = Counter()
    for w in tensor_types['weights_4d']:
        # Normalize shape description
        h, w_size, c_in, c_out = w['shape']
        if c_out == 1:  # Depthwise
            desc = f"DW {h}x{w_size}x{c_in}"
        else:
            desc = f"Conv {h}x{w_size} in:{c_in} out:{c_out}"
        weight_shapes[desc] += 1

    print(f"\nWeight tensor shape patterns:")
    for shape, count in sorted(weight_shapes.items()):
        print(f"  {shape}: {count}")

    # Show first and last few tensor names
    print(f"\nFirst 5 tensor names:")
    for t in tensors[:5]:
        print(f"  {t['name'][:80]}")

    print(f"\nLast 5 tensor names:")
    for t in tensors[-5:]:
        print(f"  {t['name'][:80]}")

    return tensor_types

def main():
    print("Comparing model structures...")

    st_tensors = analyze_model_ops(ST_MODEL, "ST per-tensor (WORKS)")
    my_tensors = analyze_model_ops(MY_MODEL, "My per-tensor (BROKEN)")

    # Compare weight shapes
    print(f"\n{'='*70}")
    print("WEIGHT TENSOR COMPARISON")
    print(f"{'='*70}")

    st_shapes = [tuple(w['shape']) for w in st_tensors['weights_4d']]
    my_shapes = [tuple(w['shape']) for w in my_tensors['weights_4d']]

    # Check unique shapes
    st_unique = set(st_shapes)
    my_unique = set(my_shapes)

    print(f"\nST unique shapes: {len(st_unique)}")
    print(f"My unique shapes: {len(my_unique)}")

    # Shapes in ST but not in mine
    only_st = st_unique - my_unique
    if only_st:
        print(f"\nShapes only in ST model:")
        for s in sorted(only_st):
            print(f"  {s}")

    # Shapes in mine but not ST
    only_my = my_unique - st_unique
    if only_my:
        print(f"\nShapes only in My model:")
        for s in sorted(only_my):
            print(f"  {s}")

if __name__ == "__main__":
    main()
