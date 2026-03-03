#!/usr/bin/env python3
"""
ONNX Quantization Path for MobileNetV2 and other models.

Strategy: Export Keras model -> ONNX -> Quantize with ONNX Runtime (per-tensor)
-> Convert quantized ONNX to TFLite (or deploy ONNX directly via stedgeai)

This bypasses TFLite's aggressive op fusion that breaks per-tensor quantization.
"""
import os
import sys
import shutil
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from pathlib import Path
from collections import Counter

# ONNX imports
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
CALIBRATION_FILE = DATA_DIR / "calibration.npz"
TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"
LABELS_FILE = MODELS_DIR / "ImageNetLabels.txt"
OUTPUT_DIR = Path(__file__).parent / "candidate-models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_labels():
    with open(LABELS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]


class ImageNetCalibrationReader(CalibrationDataReader):
    """Calibration data reader for ONNX Runtime quantization."""

    def __init__(self, calibration_file, input_name, preprocess_fn=None, num_samples=200):
        with np.load(str(calibration_file)) as data:
            images = data[list(data.keys())[0]]  # raw [0,255] float32
        self.images = images[:num_samples]
        self.input_name = input_name
        self.preprocess_fn = preprocess_fn
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        img = self.images[self.index:self.index + 1].astype(np.float32)
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
        self.index += 1
        return {self.input_name: img}

    def rewind(self):
        self.index = 0


def keras_to_onnx(model, output_path, model_name="model"):
    """Export a Keras model to ONNX format via tf2onnx."""
    import tf2onnx

    # Save as SavedModel first
    tmp_dir = tempfile.mkdtemp()
    saved_model_path = os.path.join(tmp_dir, "saved_model")
    model.export(saved_model_path)

    print(f"  Converting SavedModel to ONNX...")
    # Use tf2onnx command-line approach via Python API
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_path,
        "--output", str(output_path),
        "--opset", "13",
    ], capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  tf2onnx stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"tf2onnx conversion failed: {result.stderr[-200:]}")

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  ONNX model saved to: {output_path}")
    return output_path


def quantize_onnx_per_tensor(onnx_model_path, output_path, preprocess_fn=None):
    """Quantize an ONNX model with per-tensor (per-layer) quantization."""

    # Load the model to get input name
    model = onnx.load(str(onnx_model_path))
    input_name = model.graph.input[0].name
    print(f"  ONNX input name: {input_name}")

    # Create calibration reader
    calibration_reader = ImageNetCalibrationReader(
        CALIBRATION_FILE, input_name, preprocess_fn, num_samples=200
    )

    print(f"  Quantizing ONNX model per-tensor...")
    quantize_static(
        model_input=str(onnx_model_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QOperator,  # QOperator = quantized ops (vs QDQ)
        per_channel=False,  # per-tensor!
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"  Quantized ONNX saved to: {output_path}")
    return output_path


def run_onnx_inference(model_path, image_path):
    """Run inference on an ONNX model."""
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type

    print(f"  ONNX input: name={input_name}, shape={input_shape}, type={input_type}")

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img, dtype=np.float32)

    # Check if input expects uint8
    if 'uint8' in input_type:
        img_np = np.array(img, dtype=np.uint8)
    # Keep as [0,255] float32 for EfficientNet-style models

    tensor = np.expand_dims(img_np, 0)
    outputs = session.run(None, {input_name: tensor})
    output = outputs[0][0]

    top5_idx = output.argsort()[-5:][::-1]
    top5_conf = output[top5_idx]

    return top5_idx, top5_conf, output


def run_tflite_inference(model_path, image_path):
    """Run inference on a TFLite model."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
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

    return top5_idx, top5_conf, output


def analyze_onnx_ops(model_path):
    """Analyze operations in an ONNX model."""
    model = onnx.load(str(model_path))
    op_counts = Counter()
    for node in model.graph.node:
        op_counts[node.op_type] += 1
    return dict(op_counts)


def test_model_onnx_path(model_name, build_fn, preprocess_fn, labels):
    """Test a model through the ONNX quantization path."""
    print(f"\n{'=' * 80}")
    print(f"ONNX PATH: {model_name}")
    print(f"{'=' * 80}")

    try:
        # Step 1: Build model
        print(f"  Step 1: Building Keras model...")
        model = build_fn()
        print(f"  Parameters: {model.count_params():,}")

        # Step 2: Export to ONNX
        onnx_fp32_path = OUTPUT_DIR / f"{model_name.lower()}_fp32.onnx"
        print(f"  Step 2: Exporting to ONNX...")
        keras_to_onnx(model, onnx_fp32_path, model_name)
        fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
        print(f"  FP32 ONNX size: {fp32_size:.1f} MB")

        # Verify FP32 ONNX inference
        print(f"  Verifying FP32 ONNX inference...")
        top5_idx, top5_conf, _ = run_onnx_inference(onnx_fp32_path, TEST_IMAGE)
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            print(f"    #{i + 1}: {conf:.4f} [{idx}] {label}")

        fp32_correct = any(
            idx < len(labels) and 'water' in labels[idx].lower()
            for idx in top5_idx
        )
        print(f"  FP32 Result: {'✅ PASS' if fp32_correct else '❌ FAIL'}")

        if not fp32_correct:
            print(f"  ⚠️ FP32 model already fails - check preprocessing!")

        # Analyze FP32 ops
        print(f"  FP32 ONNX ops:")
        ops = analyze_onnx_ops(onnx_fp32_path)
        for op, count in sorted(ops.items(), key=lambda x: -x[1])[:15]:
            print(f"    {op}: {count}")

        # Step 3: Quantize with ONNX Runtime
        onnx_int8_path = OUTPUT_DIR / f"{model_name.lower()}_int8_pt.onnx"
        print(f"  Step 3: Quantizing per-tensor with ONNX Runtime...")
        quantize_onnx_per_tensor(onnx_fp32_path, onnx_int8_path, preprocess_fn)
        int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
        print(f"  INT8 ONNX size: {int8_size:.1f} MB")

        # Step 4: Verify quantized inference
        print(f"  Step 4: Verifying quantized ONNX inference...")
        top5_idx, top5_conf, _ = run_onnx_inference(onnx_int8_path, TEST_IMAGE)
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            print(f"    #{i + 1}: {conf:.4f} [{idx}] {label}")

        int8_correct = any(
            idx < len(labels) and 'water' in labels[idx].lower()
            for idx in top5_idx
        )

        # Analyze quantized ops
        print(f"  Quantized ONNX ops:")
        q_ops = analyze_onnx_ops(onnx_int8_path)
        for op, count in sorted(q_ops.items(), key=lambda x: -x[1])[:15]:
            print(f"    {op}: {count}")

        if int8_correct:
            print(f"\n  ✅ SUCCESS! {model_name} per-tensor via ONNX path WORKS!")
            print(f"  Quantized model: {onnx_int8_path}")
            print(f"  Deploy with: stedgeai generate -m {onnx_int8_path} --target stm32mp25")
        else:
            print(f"\n  ❌ FAIL - {model_name} per-tensor via ONNX path does not classify correctly")

        return int8_correct, str(onnx_int8_path)

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_tflite_direct(model_name, tflite_path, labels):
    """Test inference on an existing TFLite model."""
    print(f"\n{'=' * 80}")
    print(f"TFLITE DIRECT: {model_name}")
    print(f"{'=' * 80}")

    if not os.path.exists(tflite_path):
        print(f"  File not found: {tflite_path}")
        return False

    top5_idx, top5_conf, _ = run_tflite_inference(tflite_path, TEST_IMAGE)
    for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"    #{i + 1}: {conf:.4f} [{idx}] {label}")

    correct = any(
        idx < len(labels) and 'water' in labels[idx].lower()
        for idx in top5_idx
    )
    print(f"  Result: {'✅ PASS' if correct else '❌ FAIL'}")
    return correct


def main():
    print("=" * 80)
    print("ONNX QUANTIZATION PATH - Bypass TFLite Converter Fusion")
    print("=" * 80)

    labels = load_labels()
    print(f"Loaded {len(labels)} labels")

    models_to_test = {}

    # Priority 1: MobileNetV2 - the ideal model if we can get per-tensor working
    def build_mv2():
        return keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    models_to_test['mobilenetv2'] = (build_mv2, None)

    # Priority 2: MobileNetV1 - simpler, fast on NPU
    def build_mv1():
        return keras.applications.MobileNet(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    models_to_test['mobilenetv1'] = (build_mv1, None)

    # Priority 3: ResNet50 - has perfect NPU op profile (97% NPU, 0% GPU)
    def build_resnet50():
        return keras.applications.ResNet50(
            input_shape=(224, 224, 3), include_top=True, weights='imagenet'
        )
    models_to_test['resnet50'] = (build_resnet50, None)

    # Allow selecting specific models
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        models_to_test = {k: v for k, v in models_to_test.items() if k in selected}

    results = []
    for name, (build_fn, preprocess_fn) in models_to_test.items():
        success, model_path = test_model_onnx_path(name, build_fn, preprocess_fn, labels)
        results.append((name, success, model_path))
        tf.keras.backend.clear_session()

    # Also test existing TFLite models for reference
    print(f"\n\n{'=' * 80}")
    print("REFERENCE: Existing TFLite Models")
    print(f"{'=' * 80}")
    test_tflite_direct("ST MobileNetV2 PT", MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite", labels)
    effnet_pt = OUTPUT_DIR / "efficientnetv2b0_pt.tflite"
    if effnet_pt.exists():
        test_tflite_direct("EfficientNetV2B0 PT", effnet_pt, labels)

    # Summary
    print(f"\n\n{'=' * 80}")
    print("ONNX PATH SUMMARY")
    print(f"{'=' * 80}")
    for name, success, path in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status} -> {path}")

    passing = [(n, p) for n, s, p in results if s]
    if passing:
        print(f"\n🎉 WORKING MODELS via ONNX path:")
        for name, path in passing:
            print(f"  {name}: {path}")
            print(f"  Deploy: stedgeai generate -m {path} --target stm32mp25")
    else:
        print(f"\n  No models passed via ONNX path either.")
        print(f"  The per-tensor quantization precision loss may be fundamental,")
        print(f"  not just a TFLite converter issue.")


if __name__ == '__main__':
    main()

