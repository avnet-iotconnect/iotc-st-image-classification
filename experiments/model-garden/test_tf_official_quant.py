#!/usr/bin/env python3
"""
Download and test TensorFlow's official pre-quantized MobileNetV2 model.
This model may have been created with older tools that don't fuse operations.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import urllib.request
import tarfile
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"

# TensorFlow's official pre-quantized models
TF_MODELS = {
    "tf_mobilenet_v2_quant": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224_quant.tgz",
    "tf_mobilenet_v1_quant": "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_quant.tgz",
}

def download_and_extract(url, output_dir):
    """Download and extract tgz file"""
    tgz_path = output_dir / "model.tgz"
    print(f"  Downloading from {url}...")
    urllib.request.urlretrieve(url, tgz_path)

    print(f"  Extracting...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(output_dir)

    # Find the tflite file
    tflite_files = list(output_dir.glob("**/*.tflite"))
    if tflite_files:
        return tflite_files[0]
    return None

def count_ops(model_path):
    """Count operations in model"""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    return dict(Counter(op.get('op_name', 'unknown') for op in ops))

def test_inference(model_path, image_path, labels, num_classes=1001):
    """Run inference"""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

    # Check input type
    if in_details['dtype'] == np.uint8:
        tensor = img_np.astype(np.uint8)
    else:
        tensor = (img_np.astype(np.float32) / 127.5) - 1.0

    tensor = np.expand_dims(tensor, 0)
    interp.set_tensor(in_details['index'], tensor)
    interp.invoke()

    output = interp.get_tensor(out_details['index'])[0]
    top5_idx = output.argsort()[-5:][::-1]
    top5_conf = output[top5_idx]

    # Normalize if uint8
    if out_details['dtype'] == np.uint8:
        top5_conf = top5_conf.astype(np.float32) / 255.0

    return top5_idx, top5_conf

def main():
    # Write output to file
    log_path = OUTPUT_DIR / "tf_official_models_test.txt"

    with open(log_path, 'w') as log:
        def write(msg):
            print(msg)
            log.write(msg + '\n')
            log.flush()

        write("="*70)
        write("Testing TensorFlow Official Pre-Quantized Models")
        write("="*70)

        # Load labels
        labels_path = MODELS_DIR / "ImageNetLabels.txt"
        with open(labels_path) as f:
            labels = [line.strip() for line in f.readlines()]

        # Test each TF official model
        for name, url in TF_MODELS.items():
            write(f"\n{'='*70}")
            write(f"Model: {name}")
            write(f"{'='*70}")

            try:
                model_dir = OUTPUT_DIR / name
                model_dir.mkdir(exist_ok=True)

                tflite_path = download_and_extract(url, model_dir)
                if not tflite_path:
                    write("  ERROR: No tflite file found")
                    continue

                write(f"  Model path: {tflite_path}")

                # Count ops
                ops = count_ops(str(tflite_path))
                write(f"\n  Operations: {ops}")
                write(f"  Has MINIMUM: {ops.get('MINIMUM', 0)}")
                write(f"  Has RELU: {ops.get('RELU', 0)}")
                write(f"  Has RELU6: {ops.get('RELU6', 0)}")

                # Check if this is unfused
                is_unfused = ops.get('MINIMUM', 0) > 0 or ops.get('RELU', 0) > 0 or ops.get('RELU6', 0) > 0
                write(f"  Unfused activations: {'YES ✓' if is_unfused else 'NO ✗'}")

                # Test inference
                write(f"\n  Inference on water bottle image:")
                top5_idx, top5_conf = test_inference(str(tflite_path), TEST_IMAGE, labels)
                for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
                    label = labels[idx] if idx < len(labels) else f"idx_{idx}"
                    marker = " <<<" if i == 0 else ""
                    write(f"    #{i+1}: {conf:.4f} [{idx:4d}] {label}{marker}")

            except Exception as e:
                write(f"  ERROR: {e}")

        # Compare with ST model
        write(f"\n{'='*70}")
        write("Reference: ST Model")
        write(f"{'='*70}")

        st_path = MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite"
        st_ops = count_ops(str(st_path))
        write(f"  Operations: {st_ops}")
        write(f"  Has MINIMUM: {st_ops.get('MINIMUM', 0)}")
        write(f"  Has RELU: {st_ops.get('RELU', 0)}")

        write(f"\n  Inference on water bottle image:")
        top5_idx, top5_conf = test_inference(str(st_path), TEST_IMAGE, labels)
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            label = labels[idx] if idx < len(labels) else f"idx_{idx}"
            write(f"    #{i+1}: {conf:.4f} [{idx:4d}] {label}")

        write(f"\nResults written to: {log_path}")

if __name__ == "__main__":
    main()
