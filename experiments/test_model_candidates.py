#!/usr/bin/env python3
"""
Test multiple model architectures for per-tensor quantization suitability.
Goals:
  1. Check which models produce correct inference after per-tensor quantization
  2. Analyze TFLite op profiles to predict NPU compatibility
  3. Find the best replacement for EfficientNetV2B0 (which works but is slow)

Key NPU-friendly ops (VIPNano-SI): CONV_2D, DEPTHWISE_CONV_2D, RELU, RELU6,
    ADD, AVERAGE_POOL_2D, MAX_POOL_2D, CONCATENATION, RESHAPE, SOFTMAX, PAD

Problematic ops (fall to GPU): LOGISTIC, MUL, HARD_SWISH, EXP, RSQRT, SUB
"""
import os
import sys
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
CALIBRATION_FILE = DATA_DIR / "calibration.npz"
TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"
LABELS_FILE = MODELS_DIR / "ImageNetLabels.txt"
OUTPUT_DIR = Path(__file__).parent / "candidate-models"

# Ops that the ST NPU handles natively
NPU_FRIENDLY_OPS = {
    'CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED',
    'AVERAGE_POOL_2D', 'MAX_POOL_2D',
    'ADD', 'CONCATENATION',
    'RELU', 'RELU6',
    'RESHAPE', 'SOFTMAX', 'PAD',
    'QUANTIZE', 'DEQUANTIZE',
    'MEAN',  # global average pooling
    'SQUEEZE',
    'MINIMUM',  # used in unfused ReLU6
}

# Ops that may fall to GPU
GPU_FALLBACK_OPS = {
    'LOGISTIC', 'MUL', 'HARD_SWISH', 'EXP', 'RSQRT', 'SUB',
    'TANH', 'DIV', 'FLOOR', 'CEIL', 'GATHER', 'SLICE',
    'STRIDED_SLICE', 'TRANSPOSE', 'SPLIT', 'SPLIT_V',
}


def load_labels():
    with open(LABELS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]


def representative_dataset_gen():
    with np.load(str(CALIBRATION_FILE)) as data:
        images = data[list(data.keys())[0]]
        for i in range(min(len(images), 200)):
            yield [images[i:i + 1]]


def quantize_per_tensor(model, preprocess_fn=None):
    """Quantize a model with per-tensor quantization."""

    def rep_dataset():
        with np.load(str(CALIBRATION_FILE)) as data:
            images = data[list(data.keys())[0]]  # raw [0,255] float32
            for i in range(min(len(images), 200)):
                img = images[i:i + 1]
                if preprocess_fn is not None:
                    img = preprocess_fn(img)
                yield [img.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter.representative_dataset = rep_dataset
    converter._experimental_disable_per_channel = True
    return converter.convert()


def analyze_tflite_ops(tflite_model_bytes):
    """Analyze ops in a TFLite model buffer."""
    interp = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interp.allocate_tensors()
    ops = interp._get_ops_details()

    op_counts = Counter()
    npu_ops = 0
    gpu_ops = 0
    unknown_ops = []

    for op in ops:
        name = op.get('op_name', 'unknown')
        op_counts[name] += 1
        if name in NPU_FRIENDLY_OPS:
            npu_ops += 1
        elif name in GPU_FALLBACK_OPS:
            gpu_ops += 1
        else:
            unknown_ops.append(name)

    total = len(ops)
    return {
        'total_ops': total,
        'op_counts': dict(op_counts),
        'npu_ops': npu_ops,
        'gpu_ops': gpu_ops,
        'npu_pct': (npu_ops / total * 100) if total > 0 else 0,
        'gpu_pct': (gpu_ops / total * 100) if total > 0 else 0,
        'unknown_ops': list(set(unknown_ops)),
        'has_logistic': op_counts.get('LOGISTIC', 0) > 0,
        'has_mul': op_counts.get('MUL', 0) > 0,
        'has_hard_swish': op_counts.get('HARD_SWISH', 0) > 0,
    }


def run_inference_tflite(tflite_model_bytes, image_path, num_classes=1000):
    """Run inference on a TFLite model buffer."""
    interp = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

    in_dtype = in_details['dtype']
    if in_dtype == np.uint8:
        tensor = img_np.astype(np.uint8)
    elif in_dtype == np.int8:
        tensor = (img_np.astype(np.int32) - 128).astype(np.int8)
    else:
        tensor = img_np.astype(np.float32)

    tensor = np.expand_dims(tensor, 0)
    interp.set_tensor(in_details['index'], tensor)
    interp.invoke()

    output = interp.get_tensor(out_details['index'])[0]
    top5_idx = output.argsort()[-5:][::-1]
    top5_conf = output[top5_idx]

    return top5_idx, top5_conf


def get_model_candidates():
    """Return a dict of model name -> (build_fn, preprocess_fn, notes)"""
    candidates = {}

    # 1. MobileNetV1 - uses ReLU6 but simpler architecture
    def build_mobilenetv1():
        return keras.applications.MobileNet(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    candidates['MobileNetV1'] = (build_mobilenetv1, None, 'ReLU6, no inverted residuals')

    # 2. MobileNetV2 (baseline - known to fail)
    def build_mobilenetv2():
        return keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    candidates['MobileNetV2'] = (build_mobilenetv2, None, 'KNOWN FAIL - baseline')

    # 3. EfficientNetV2B0 (current working model - known slow)
    def build_efficientnetv2b0():
        return keras.applications.EfficientNetV2B0(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    candidates['EfficientNetV2B0'] = (build_efficientnetv2b0, None, 'CURRENT - works but slow 27ms')

    # 4. NASNetMobile - ReLU activations, separable convolutions
    def build_nasnetmobile():
        return keras.applications.NASNetMobile(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    # NASNetMobile uses different input preprocessing
    candidates['NASNetMobile'] = (build_nasnetmobile, None, 'ReLU, separable convs, complex cell')

    # 5. DenseNet121 - ReLU activations
    def build_densenet121():
        return keras.applications.DenseNet121(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    candidates['DenseNet121'] = (build_densenet121, None, 'ReLU, standard convs, concat-heavy')

    # 6. ResNet50 - plain ReLU, well-studied quantization
    def build_resnet50():
        return keras.applications.ResNet50(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    candidates['ResNet50'] = (build_resnet50, None, 'ReLU, large model, proven quantization')

    # 7. ResNet50V2 - uses pre-activation (BN-ReLU before conv)
    def build_resnet50v2():
        return keras.applications.ResNet50V2(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    candidates['ResNet50V2'] = (build_resnet50v2, None, 'Pre-activation ResNet, ReLU')

    # 8. InceptionV3 - ReLU, complex but proven
    def build_inceptionv3():
        return keras.applications.InceptionV3(
            input_shape=(299, 299, 3), include_top=True, weights='imagenet'
        )
    candidates['InceptionV3'] = (build_inceptionv3, None, 'ReLU, 299x299 input, large')

    # 9. Xception - ReLU, separable convolutions
    def build_xception():
        return keras.applications.Xception(
            input_shape=(299, 299, 3), include_top=True, weights='imagenet'
        )
    candidates['Xception'] = (build_xception, None, 'ReLU, separable convs, 299x299')

    # 10. MobileNetV2 with unfused ops (custom build)
    def build_mobilenetv2_unfused():
        return build_unfused_mobilenetv2()
    candidates['MobileNetV2-Unfused'] = (build_mobilenetv2_unfused, None, 'EXPERIMENTAL: explicit clip_by_value')

    # 11. ConvNeXtTiny - uses GeLU, may or may not work
    # Skip - GeLU likely has same GPU fallback issue

    # 12. EfficientNetB0 (v1) - uses Swish like v2
    def build_efficientnetb0():
        return keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    candidates['EfficientNetB0'] = (build_efficientnetb0, None, 'Swish activation, like V2')

    return candidates


def build_unfused_mobilenetv2():
    """
    Build MobileNetV2 with explicit unfused ReLU6 activations.
    Instead of passing activation='relu6' to Conv2D layers,
    we use a separate ReLU(max_value=6) layer so the TFLite converter
    cannot fuse them.
    """
    from keras import layers, Model

    # Load the standard MobileNetV2 to get weights
    source_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=True, weights='imagenet'
    )

    # Strategy: intercept the model and rebuild with unfused activations
    # We'll do this by modifying the model's graph

    # Actually, let's try a simpler approach:
    # Build MobileNetV2 with linear activations, then add ReLU6 as separate layers

    # First, let's see if we can just use a wrapper that adds clip_by_value
    class ExplicitReLU6(layers.Layer):
        def call(self, x):
            return tf.clip_by_value(x, 0.0, 6.0)

    # Build the functional model by walking through the source model
    # and replacing fused relu6 activations with separate ops
    input_tensor = layers.Input(shape=(224, 224, 3), name='input_unfused')

    # This is complex - instead, let's try loading and modifying the saved model
    # by replacing activations in config

    model_config = source_model.get_config()

    # Modify config to remove fused activations
    def remove_fused_activations(config):
        """Recursively modify layer configs to remove fused activations"""
        if isinstance(config, dict):
            if 'activation' in config and config['activation'] in ['relu6', 'relu']:
                original = config['activation']
                config['activation'] = 'linear'  # Remove fusion
                return original
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    remove_fused_activations(value)
        elif isinstance(config, list):
            for item in config:
                if isinstance(item, (dict, list)):
                    remove_fused_activations(item)
        return None

    # This approach is fragile, let's try something different
    # Use the source model but wrap it
    print("  Building unfused MobileNetV2 (experimental)...")

    # Method: Build MobileNetV2 without top, then reconstruct with unfused ops
    # Actually simplest: use tf.saved_model with concrete functions

    # Let's try yet another approach: clone the model with modified forward pass
    # by using a custom clone function

    # Simplest viable approach: Export to SavedModel, re-import, convert
    # with experimental flags to prevent fusion

    # Actually, the simplest test is: try converting with the MLIR-based converter
    # which has different fusion behavior

    # For now, just return the standard model - we'll test if converter flags help
    print("  NOTE: Using standard MobileNetV2 for now (unfused approach TBD)")
    return source_model


def test_model(name, build_fn, preprocess_fn, notes, labels):
    """Test a single model candidate."""
    print(f"\n{'=' * 80}")
    print(f"TESTING: {name}")
    print(f"Notes: {notes}")
    print(f"{'=' * 80}")

    result = {
        'name': name,
        'notes': notes,
        'status': 'UNKNOWN',
        'error': None,
        'ops': None,
        'top1_label': None,
        'top1_conf': None,
        'correct': False,
    }

    try:
        # Build model
        print(f"  Building model...")
        model = build_fn()
        param_count = model.count_params()
        print(f"  Parameters: {param_count:,}")
        result['params'] = param_count

        # Check input shape
        input_shape = model.input_shape
        print(f"  Input shape: {input_shape}")
        result['input_shape'] = str(input_shape)

        # Quantize
        print(f"  Quantizing per-tensor...")
        tflite_bytes = quantize_per_tensor(model, preprocess_fn)
        model_size_mb = len(tflite_bytes) / (1024 * 1024)
        print(f"  Model size: {model_size_mb:.1f} MB")
        result['size_mb'] = model_size_mb

        # Save model
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_pt.tflite"
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)
        print(f"  Saved to: {out_path}")
        result['output_path'] = str(out_path)

        # Analyze ops
        print(f"  Analyzing operations...")
        ops = analyze_tflite_ops(tflite_bytes)
        result['ops'] = ops
        print(f"  Total ops: {ops['total_ops']}")
        print(f"  NPU-friendly: {ops['npu_ops']} ({ops['npu_pct']:.1f}%)")
        print(f"  GPU-fallback:  {ops['gpu_ops']} ({ops['gpu_pct']:.1f}%)")
        if ops['unknown_ops']:
            print(f"  Unknown ops: {ops['unknown_ops']}")
        print(f"  Op breakdown:")
        for op, count in sorted(ops['op_counts'].items(), key=lambda x: -x[1]):
            marker = ""
            if op in GPU_FALLBACK_OPS:
                marker = " ⚠️  GPU"
            elif op in NPU_FRIENDLY_OPS:
                marker = " ✓ NPU"
            print(f"    {op}: {count}{marker}")

        # Run inference
        print(f"  Running inference on test image...")
        # Handle different input sizes
        if '299' in str(model.input_shape):
            print(f"  WARNING: Model uses 299x299 input, skipping standard inference test")
            result['status'] = 'QUANTIZED_OK'
            result['top1_label'] = 'N/A (wrong input size)'
            return result

        top5_idx, top5_conf = run_inference_tflite(tflite_bytes, TEST_IMAGE)

        # Check if output uses 1001 classes (TFHub style) or 1000
        num_classes = len(top5_idx)

        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            print(f"    #{i + 1}: {conf:.4f} [{idx}] {label}")

        top1_label = labels[top5_idx[0]] if top5_idx[0] < len(labels) else f"class_{top5_idx[0]}"
        result['top1_label'] = top1_label
        result['top1_conf'] = float(top5_conf[0])

        # Check if water bottle is in top-5
        water_bottle_found = False
        for idx in top5_idx:
            if idx < len(labels) and 'water' in labels[idx].lower():
                water_bottle_found = True
                break

        result['correct'] = water_bottle_found
        if water_bottle_found:
            result['status'] = 'PASS ✅'
            print(f"  RESULT: ✅ PASS - Water bottle detected!")
        else:
            result['status'] = 'FAIL ❌'
            print(f"  RESULT: ❌ FAIL - Water bottle NOT in top-5")

    except Exception as e:
        result['status'] = f'ERROR ❌'
        result['error'] = str(e)
        print(f"  ERROR: {e}")
        traceback.print_exc()

    return result


def main():
    print("=" * 80)
    print("MODEL CANDIDATE EVALUATION FOR ST MP2 NPU")
    print("Per-tensor quantization + NPU op compatibility")
    print("=" * 80)

    labels = load_labels()
    print(f"Loaded {len(labels)} labels")

    # Check calibration data exists
    if not CALIBRATION_FILE.exists():
        print(f"ERROR: Calibration file not found: {CALIBRATION_FILE}")
        sys.exit(1)

    # Check what preprocessing the calibration data uses
    with np.load(str(CALIBRATION_FILE)) as data:
        cal_images = data[list(data.keys())[0]]
        print(f"Calibration data: {cal_images.shape}, range [{cal_images.min():.1f}, {cal_images.max():.1f}]")

    candidates = get_model_candidates()

    # Allow selecting specific models from command line
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        candidates = {k: v for k, v in candidates.items() if k in selected}
        if not candidates:
            print(f"No matching models. Available: {list(get_model_candidates().keys())}")
            sys.exit(1)

    results = []
    for name, (build_fn, preprocess_fn, notes) in candidates.items():
        result = test_model(name, build_fn, preprocess_fn, notes, labels)
        results.append(result)
        # Free memory
        tf.keras.backend.clear_session()

    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<25} {'Status':<15} {'Size(MB)':<10} {'NPU%':<8} {'GPU%':<8} {'Top-1':<25} {'GPU Ops'}")
    print("-" * 110)
    for r in results:
        ops = r.get('ops', {})
        npu_pct = f"{ops.get('npu_pct', 0):.0f}%" if ops else "N/A"
        gpu_pct = f"{ops.get('gpu_pct', 0):.0f}%" if ops else "N/A"
        size = f"{r.get('size_mb', 0):.1f}" if 'size_mb' in r else "N/A"
        top1 = r.get('top1_label', 'N/A')
        if len(top1) > 24:
            top1 = top1[:21] + "..."
        gpu_ops = ""
        if ops and ops.get('op_counts'):
            problematic = {k: v for k, v in ops['op_counts'].items() if k in GPU_FALLBACK_OPS}
            gpu_ops = ", ".join(f"{k}:{v}" for k, v in problematic.items())
        print(f"{r['name']:<25} {r.get('status', 'N/A'):<15} {size:<10} {npu_pct:<8} {gpu_pct:<8} {top1:<25} {gpu_ops}")

    # Recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 80}")
    passing = [r for r in results if r.get('correct', False)]
    if passing:
        # Sort by NPU percentage (higher is better), then by model size (smaller is better)
        passing.sort(key=lambda r: (
            -r.get('ops', {}).get('npu_pct', 0),
            r.get('size_mb', 999)
        ))
        print("\nModels that PASSED per-tensor quantization, ranked by NPU suitability:")
        for i, r in enumerate(passing, 1):
            ops = r.get('ops', {})
            gpu_problematic = {k: v for k, v in ops.get('op_counts', {}).items() if k in GPU_FALLBACK_OPS}
            print(f"\n  {i}. {r['name']}")
            print(f"     Size: {r.get('size_mb', 'N/A')} MB")
            print(f"     NPU ops: {ops.get('npu_pct', 'N/A'):.1f}%")
            print(f"     GPU fallback ops: {gpu_problematic if gpu_problematic else 'NONE'}")
            if not gpu_problematic:
                print(f"     ⭐ ALL OPS NPU-FRIENDLY - should match MobileNetV2 NPU utilization!")
    else:
        print("\nNo models passed. Consider:")
        print("  1. ONNX quantization path (bypass TFLite converter fusion)")
        print("  2. Custom unfused MobileNetV2")
        print("  3. TF1 environment")


if __name__ == '__main__':
    main()

