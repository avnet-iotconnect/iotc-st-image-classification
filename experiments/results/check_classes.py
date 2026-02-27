#!/usr/bin/env python3
"""
Check model output dimensions and class count
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

MODELS = {
    "ST per-tensor": MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite",
    "My per-tensor": MODELS_DIR / "quantized-pt.tflite",
    "My per-channel": MODELS_DIR / "quantized-pc.tflite",
}

for name, path in MODELS.items():
    if not path.exists():
        continue

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    print(f"{name}:")
    print(f"  Input: shape={inp['shape']} dtype={inp['dtype']}")
    print(f"  Output: shape={out['shape']} dtype={out['dtype']}")
    print(f"  Num classes: {out['shape'][-1]}")
    print()
