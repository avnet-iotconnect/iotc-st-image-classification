#!/usr/bin/env python3
"""
Quick candidate model test: quantize to per-tensor TFLite + analyze ops.
Models chosen for NPU-friendly op profiles (ReLU-only, no Swish/sigmoid).

After this script runs, copy the .tflite files to the board and convert:
  scp experiments/candidate-models/*.tflite root@192.168.38.141:app/
  ssh root@192.168.38.141
  cd app
  for f in *.tflite; do
    name="${f%.tflite}"
    stedgeai generate -m "$f" --target stm32mp25
    mv stm32ai_output/"$name".nb . 2>/dev/null
    rm -rf stm32ai_output stm32ai_ws
  done
  x-linux-ai-benchmark -m app/MODEL.nb
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
CALIBRATION_FILE = DATA_DIR / "calibration.npz"
OUTPUT_DIR = Path(__file__).parent / "candidate-models"
OUTPUT_DIR.mkdir(exist_ok=True)

# Ops the VIPNano-SI NPU handles well
NPU_OPS = {'CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED','AVERAGE_POOL_2D',
            'MAX_POOL_2D','ADD','CONCATENATION','RELU','RELU6','RESHAPE',
            'SOFTMAX','PAD','QUANTIZE','DEQUANTIZE','MEAN','SQUEEZE','MINIMUM'}
GPU_OPS = {'LOGISTIC','MUL','HARD_SWISH','EXP','RSQRT','SUB','TANH','DIV',
           'STRIDED_SLICE','TRANSPOSE','SPLIT','SPLIT_V','GATHER'}


def quantize_per_tensor(model):
    """Per-tensor quantize using real calibration data."""
    def rep_data():
        with np.load(str(CALIBRATION_FILE)) as d:
            imgs = d[list(d.keys())[0]]  # [0,255] float32
            for i in range(min(len(imgs), 200)):
                yield [imgs[i:i+1]]

    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.uint8
    c.inference_output_type = tf.float32
    c.representative_dataset = rep_data
    c._experimental_disable_per_channel = True
    return c.convert()


def analyze_ops(tflite_bytes):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    counts = Counter(op.get('op_name','?') for op in ops)
    npu = sum(v for k,v in counts.items() if k in NPU_OPS)
    gpu = sum(v for k,v in counts.items() if k in GPU_OPS)
    total = len(ops)
    return counts, npu, gpu, total


def test_inference(tflite_bytes, label=""):
    """Quick inference on test image to check if model produces reasonable output."""
    from PIL import Image
    LABELS = Path(__file__).parent.parent / "models" / "ImageNetLabels.txt"
    TEST_IMG = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"

    labels = open(LABELS).read().strip().split('\n')
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ind = interp.get_input_details()[0]
    oud = interp.get_output_details()[0]

    img = Image.open(TEST_IMG).convert("RGB").resize((224,224))
    arr = np.array(img)
    if ind['dtype'] == np.uint8:
        t = arr.astype(np.uint8)
    else:
        t = arr.astype(np.float32)
    t = np.expand_dims(t, 0)
    interp.set_tensor(ind['index'], t)
    interp.invoke()
    out = interp.get_tensor(oud['index'])[0]
    top5 = out.argsort()[-5:][::-1]

    print(f"  Inference ({label}):")
    water = False
    for i, idx in enumerate(top5):
        lbl = labels[idx] if idx < len(labels) else f"#{idx}"
        print(f"    {i+1}. [{idx}] {lbl} = {out[idx]:.4f}")
        if 'water' in lbl.lower():
            water = True
    return water


# ============================================================
# CANDIDATE MODELS
# ============================================================
# Selection criteria:
# - 224x224 input (matches our pipeline)
# - Uses ReLU (not Swish/HardSwish/SiLU/sigmoid)
# - Small enough for embedded (~3-25MB quantized)
# - Available in keras.applications
#
# WHY these specific models:
#
# MobileNetV1: Same depthwise-separable as V2 but simpler (no inverted residuals).
#   Uses ReLU6. Very fast on NPU. Per-tensor may fail (same fusion issue) but
#   worth confirming on actual NPU benchmarks even if accuracy is off.
#
# ResNet50: Uses plain ReLU, standard convs. Perfect NPU op profile (97% NPU,
#   0% GPU in our tests). Large model but very well-studied for quantization.
#   24MB quantized. May be too big/slow but great reference point.
#
# EfficientNetV2B0: Our current working model. Swish causes GPU fallback.
#   BASELINE for comparison - we know this works at 27ms.
#
# MobileNetV2: The ideal model. We know TFLite per-tensor breaks accuracy,
#   but it's worth benchmarking the NBG anyway to see if the NPU compiler
#   compensates. Sometimes NBG conversion fixes quantization issues.
#
# NASNetMobile: ReLU + separable convs. Complex cell structure but
#   all NPU-friendly ops. 5.3MB. Worth trying.
#
# DenseNet121: ReLU, but our test showed 62 MUL ops from BatchNorm.
#   Skip - too many GPU ops.
#
# EfficientNetB0 (v1): Same Swish issue as V2. Skip.
#
# InceptionV3: ReLU, but 299x299 input. Skip - wrong input size.

CANDIDATES = {
    # name: (builder_fn, notes)
    "mobilenetv1": (
        lambda: keras.applications.MobileNet(input_shape=(224,224,3), include_top=True, weights='imagenet'),
        "ReLU6, depthwise-sep, simple arch, ~4MB"
    ),
    "mobilenetv2": (
        lambda: keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=True, weights='imagenet'),
        "ReLU6, inverted residuals, ~3.5MB - known TFLite PT issue but test NBG"
    ),
    "resnet50": (
        lambda: keras.applications.ResNet50(input_shape=(224,224,3), include_top=True, weights='imagenet'),
        "Plain ReLU, standard convs, ~24MB, 97% NPU ops"
    ),
    "efficientnetv2b0": (
        lambda: keras.applications.EfficientNetV2B0(input_shape=(224,224,3), include_top=True, weights='imagenet'),
        "Swish (LOGISTIC+MUL), BASELINE 27ms/75% NPU"
    ),
    "nasnetmobile": (
        lambda: keras.applications.NASNetMobile(input_shape=(224,224,3), include_top=True, weights='imagenet'),
        "ReLU, separable convs, complex cells, ~5MB"
    ),
}


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(CANDIDATES.keys())

    results = []
    for name in selected:
        if name not in CANDIDATES:
            print(f"Unknown model: {name}. Available: {list(CANDIDATES.keys())}")
            continue

        build_fn, notes = CANDIDATES[name]
        print(f"\n{'='*70}")
        print(f"  {name}: {notes}")
        print(f"{'='*70}")

        try:
            print(f"  Building...")
            model = build_fn()
            params = model.count_params()
            print(f"  Params: {params:,}")

            print(f"  Quantizing per-tensor...")
            tfl = quantize_per_tensor(model)
            size_mb = len(tfl) / 1024 / 1024
            print(f"  Size: {size_mb:.1f} MB")

            counts, npu, gpu, total = analyze_ops(tfl)
            npu_pct = npu/total*100 if total else 0
            gpu_pct = gpu/total*100 if total else 0

            gpu_detail = {k:v for k,v in counts.items() if k in GPU_OPS}
            print(f"  Ops: {total} total, NPU={npu_pct:.0f}%, GPU={gpu_pct:.0f}%")
            if gpu_detail:
                print(f"  GPU ops: {gpu_detail}")

            # Quick accuracy check
            water = test_inference(tfl, name)

            out_path = OUTPUT_DIR / f"{name}-pt.tflite"
            with open(out_path, 'wb') as f:
                f.write(tfl)
            print(f"  Saved: {out_path}")

            results.append((name, size_mb, npu_pct, gpu_pct, gpu_detail, water, str(out_path)))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append((name, 0, 0, 0, {}, False, "ERROR"))

        tf.keras.backend.clear_session()

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"{'Model':<20} {'Size':>6} {'NPU%':>6} {'GPU%':>6} {'Accurate':>9}  GPU Ops")
    print(f"{'-'*90}")
    for name, sz, npu, gpu_p, gpu_d, water, path in results:
        acc = "✅ YES" if water else "❌ NO"
        gops = ", ".join(f"{k}:{v}" for k,v in gpu_d.items()) if gpu_d else "none"
        print(f"{name:<20} {sz:>5.1f}M {npu:>5.0f}% {gpu_p:>5.0f}% {acc:>9}  {gops}")

    print(f"\nNext steps:")
    print(f"  scp experiments/candidate-models/*-pt.tflite root@192.168.38.141:app/")
    print(f"  Then on the board, for each file:")
    print(f"    stedgeai generate -m FILE.tflite --target stm32mp25")
    print(f"    mv stm32ai_output/FILE.nb .")
    print(f"    x-linux-ai-benchmark -m FILE.nb")


if __name__ == '__main__':
    main()

