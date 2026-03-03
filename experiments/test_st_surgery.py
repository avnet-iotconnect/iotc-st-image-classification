#!/usr/bin/env python3
"""
Experiment: Apply ST's model_formatting_ptq_per_tensor surgery to MobileNetV2,
then quantize per-tensor and test if inference is "unbroken".

ST's code does:
1. BN folding
2. Zero irrelevant channels
3. Cross-layer equalization (CLE)
4. High bias absorption
5. Replace ReLU6 with adaptive per-channel clipping (ReLU + STCustomClip)

This should produce the MINIMUM+RELU op structure we saw in ST's working model.

NOTE: calibration.npz contains [0,255] range data (for EfficientNetV2).
MobileNetV2 expects [-1,1]. We generate correct calibration data inline.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from collections import Counter
from PIL import Image

# ── paths ──
DATA_DIR = Path(__file__).parent.parent / "data"
CALIBRATION_FILE = DATA_DIR / "calibration.npz"
MODELS_DIR = Path(__file__).parent.parent / "models"
LABELS_FILE = MODELS_DIR / "ImageNetLabels.txt"
TEST_IMG = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"
OUTPUT_DIR = Path(__file__).parent / "candidate-models"
OUTPUT_DIR.mkdir(exist_ok=True)

# Import ST's surgery code
sys.path.insert(0, str(Path(__file__).parent))
from st_model_zoo_code.model_formatting_ptq_per_tensor import model_formatting_ptq_per_tensor, STCustomClip

# NPU/GPU op classification
NPU_OPS = {'CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED','AVERAGE_POOL_2D',
            'MAX_POOL_2D','ADD','CONCATENATION','RELU','RELU6','RESHAPE',
            'SOFTMAX','PAD','QUANTIZE','DEQUANTIZE','MEAN','SQUEEZE','MINIMUM'}
GPU_OPS = {'LOGISTIC','MUL','HARD_SWISH','EXP','RSQRT','SUB','TANH','DIV',
           'STRIDED_SLICE','TRANSPOSE','SPLIT','SPLIT_V','GATHER'}


def get_mobilenetv2_calibration_data():
    """
    Load calibration images from calibration.npz (which is [0,255]) and
    convert to MobileNetV2's expected [-1, 1] range.
    """
    with np.load(str(CALIBRATION_FILE)) as d:
        imgs = d[list(d.keys())[0]]  # shape: (500, 224, 224, 3), range [0, 255]
    # MobileNetV2 preprocess: (x / 127.5) - 1.0 → [-1, 1]
    imgs = (imgs / 127.5) - 1.0
    print(f"  Calibration data: shape={imgs.shape}, range=[{imgs.min():.2f}, {imgs.max():.2f}]")
    return imgs


def quantize_per_tensor(model, calibration_imgs):
    """Per-tensor quantize a Keras model using provided calibration data."""
    def rep_data():
        for i in range(min(len(calibration_imgs), 200)):
            yield [calibration_imgs[i:i+1].astype(np.float32)]

    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.uint8
    c.inference_output_type = tf.float32
    c.representative_dataset = rep_data
    c._experimental_disable_per_channel = True
    return c.convert()


def analyze_ops(tflite_bytes):
    """Analyze TFLite ops and classify NPU vs GPU."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    counts = Counter(op.get('op_name', '?') for op in ops)
    npu = sum(v for k, v in counts.items() if k in NPU_OPS)
    gpu = sum(v for k, v in counts.items() if k in GPU_OPS)
    total = len(ops)
    return counts, npu, gpu, total


def test_inference(tflite_bytes, label=""):
    """Quick inference on water bottle test image."""
    labels = open(LABELS_FILE).read().strip().split('\n')
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ind = interp.get_input_details()[0]
    oud = interp.get_output_details()[0]

    img = Image.open(TEST_IMG).convert("RGB").resize((224, 224))
    arr = np.array(img)

    # Handle uint8 input (quantized model)
    if ind['dtype'] == np.uint8:
        t = arr.astype(np.uint8)
    else:
        # Float model expects [-1, 1]
        t = (arr.astype(np.float32) / 127.5) - 1.0
    t = np.expand_dims(t, 0)

    interp.set_tensor(ind['index'], t)
    interp.invoke()
    out = interp.get_tensor(oud['index'])[0]
    top5 = out.argsort()[-5:][::-1]

    print(f"\n  Inference ({label}):")
    water = False
    for i, idx in enumerate(top5):
        lbl = labels[idx] if idx < len(labels) else f"#{idx}"
        score = out[idx]
        print(f"    {i+1}. [{idx}] {lbl} = {score:.4f}")
        if 'water' in lbl.lower() or 'bottle' in lbl.lower():
            water = True
    return water


def test_float_model(model, label="float"):
    """Test the float Keras model directly for sanity."""
    img = Image.open(TEST_IMG).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 expects [-1, 1]
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr, verbose=0)
    top5 = preds[0].argsort()[-5:][::-1]
    labels = open(LABELS_FILE).read().strip().split('\n')

    print(f"\n  Float model inference ({label}):")
    for i, idx in enumerate(top5):
        # keras.applications uses 1000 classes (no background), but ImageNetLabels.txt has 1001 (index 0 = background)
        lbl_idx = idx + 1  # offset for background class
        lbl = labels[lbl_idx] if lbl_idx < len(labels) else f"#{idx}"
        print(f"    {i+1}. [{idx}] {lbl} = {preds[0][idx]:.4f}")


def main():
    print("=" * 70)
    print("  ST Surgery Experiment: MobileNetV2 Per-Tensor Quantization")
    print("=" * 70)

    # ── Step 1: Load MobileNetV2 ──
    print("\n[1] Loading MobileNetV2...")
    model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet'
    )
    print(f"  Params: {model.count_params():,}")
    print(f"  Layers: {len(model.layers)}")

    # ── Sanity check: float model ──
    print("\n[2] Sanity check: float model inference...")
    test_float_model(model, "original float MobileNetV2")

    # ── Step 3: Load calibration data with correct range ──
    print("\n[3] Loading calibration data (converting to [-1,1] for MobileNetV2)...")
    cal_data = get_mobilenetv2_calibration_data()

    # ── Step 4: Baseline - quantize WITHOUT surgery ──
    print("\n[4] Baseline: per-tensor quantization WITHOUT surgery...")
    try:
        tfl_baseline = quantize_per_tensor(model, cal_data)
        size_mb = len(tfl_baseline) / 1024 / 1024
        counts, npu, gpu, total = analyze_ops(tfl_baseline)
        print(f"  Size: {size_mb:.1f} MB, Ops: {total} total")
        print(f"  Op counts: {dict(counts)}")
        print(f"  NPU: {npu}/{total} ({npu/total*100:.0f}%), GPU: {gpu}/{total} ({gpu/total*100:.0f}%)")

        water_baseline = test_inference(tfl_baseline, "baseline NO surgery")

        out_path = OUTPUT_DIR / "mobilenetv2-pt-baseline.tflite"
        with open(out_path, 'wb') as f:
            f.write(tfl_baseline)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        water_baseline = False

    # Clear session to free memory
    tf.keras.backend.clear_session()

    # ── Step 5: Apply ST surgery ──
    print("\n[5] Applying ST model_formatting_ptq_per_tensor surgery...")
    model2 = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet'
    )

    try:
        model_optimized = model_formatting_ptq_per_tensor(model2)
        print(f"  Surgery complete!")
        print(f"  Optimized model layers: {len(model_optimized.layers)}")
        print(f"  Optimized model params: {model_optimized.count_params():,}")

        # Check what layer types are in the optimized model
        layer_types = Counter(type(l).__name__ for l in model_optimized.layers)
        print(f"  Layer types: {dict(layer_types)}")

        # ── Step 6: Test float model after surgery ──
        print("\n[6] Float model inference AFTER surgery...")
        test_float_model(model_optimized, "after ST surgery (float)")

        # ── Step 7: Quantize the surgically modified model ──
        print("\n[7] Per-tensor quantization WITH ST surgery...")
        tfl_surgery = quantize_per_tensor(model_optimized, cal_data)
        size_mb = len(tfl_surgery) / 1024 / 1024
        counts, npu, gpu, total = analyze_ops(tfl_surgery)
        print(f"  Size: {size_mb:.1f} MB, Ops: {total} total")
        print(f"  Op counts: {dict(counts)}")
        print(f"  NPU: {npu}/{total} ({npu/total*100:.0f}%), GPU: {gpu}/{total} ({gpu/total*100:.0f}%)")

        # Check for MINIMUM ops (the key indicator of ST's surgery working)
        if 'MINIMUM' in counts:
            print(f"  ✅ MINIMUM ops found: {counts['MINIMUM']} (ST-style adaptive clipping!)")
        else:
            print(f"  ⚠️  No MINIMUM ops found (surgery may not have produced expected structure)")

        water_surgery = test_inference(tfl_surgery, "WITH ST surgery")

        out_path = OUTPUT_DIR / "mobilenetv2-pt-st-surgery.tflite"
        with open(out_path, 'wb') as f:
            f.write(tfl_surgery)
        print(f"  Saved: {out_path}")

    except Exception as e:
        print(f"  ERROR during surgery: {e}")
        import traceback; traceback.print_exc()
        water_surgery = False

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Baseline (no surgery): {'✅ WORKS' if water_baseline else '❌ BROKEN'}")
    print(f"  With ST surgery:       {'✅ WORKS' if water_surgery else '❌ BROKEN'}")
    print()
    if water_surgery and not water_baseline:
        print("  🎉 ST SURGERY FIXES PER-TENSOR QUANTIZATION!")
        print()
        print("  Next steps:")
        print("    scp experiments/candidate-models/mobilenetv2-pt-st-surgery.tflite root@192.168.38.141:app/")
        print("    ssh root@192.168.38.141")
        print("    cd app")
        print("    stedgeai generate -m mobilenetv2-pt-st-surgery.tflite --target stm32mp25")
        print("    x-linux-ai-benchmark -m mobilenetv2-pt-st-surgery.nb")
    elif water_surgery and water_baseline:
        print("  Both work - surgery wasn't needed (but calibration fix may have helped)")
    elif not water_surgery and not water_baseline:
        print("  Both broken - surgery didn't fix it")
    else:
        print("  Unexpected: baseline works but surgery broke it")


if __name__ == '__main__':
    main()

