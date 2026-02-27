#!/usr/bin/env python3
"""
Test different TFLite converter settings to try to match ST's model structure
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"

def create_calibration_data():
    imagenet_val = DATA_DIR / "imagenet-val"
    images = []
    for class_dir in sorted(imagenet_val.iterdir())[:50]:
        if not class_dir.is_dir():
            continue
        class_images = list(class_dir.glob("*.JPEG"))[:2]
        for img_path in class_images:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            img_np = np.array(img, dtype=np.float32)
            img_np = (img_np / 127.5) - 1.0
            images.append(img_np)
    return np.stack(images, axis=0)

def test_inference(model_path, image_path):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.uint8)
    tensor = np.expand_dims(img_np, 0)

    interp.set_tensor(in_details['index'], tensor)
    interp.invoke()

    output = interp.get_tensor(out_details['index'])[0]
    top5_idx = output.argsort()[-5:][::-1]
    top5_conf = output[top5_idx]

    return top5_idx, top5_conf

def count_ops(model_path):
    """Count operations in model"""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    ops = interp._get_ops_details()

    from collections import Counter
    op_counts = Counter(op.get('op_name', 'unknown') for op in ops)
    return dict(op_counts)

def main():
    print("="*70)
    print("Testing TFLite Converter Settings")
    print("="*70)

    # Load labels
    labels_path = MODELS_DIR / "ImageNetLabels.txt"
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    # Create calibration data
    print("\nCreating calibration data...")
    cal_images = create_calibration_data()

    def representative_dataset():
        for i in range(len(cal_images)):
            yield [cal_images[i:i+1]]

    # Base model
    print("\nLoading base model...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=True
    )

    # Test different converter settings
    tests = [
        {
            'name': 'default_pt',
            'experimental_new_converter': True,
            'experimental_new_quantizer': True,
        },
        {
            'name': 'legacy_converter_pt',
            'experimental_new_converter': False,
            'experimental_new_quantizer': True,
        },
        {
            'name': 'legacy_quantizer_pt',
            'experimental_new_converter': True,
            'experimental_new_quantizer': False,
        },
        {
            'name': 'all_legacy_pt',
            'experimental_new_converter': False,
            'experimental_new_quantizer': False,
        },
    ]

    results = []

    for test in tests:
        print(f"\n{'='*70}")
        print(f"Test: {test['name']}")
        print(f"{'='*70}")

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
            converter._experimental_disable_per_channel = True

            # Set test-specific options
            if 'experimental_new_converter' in test:
                converter.experimental_new_converter = test['experimental_new_converter']
            if 'experimental_new_quantizer' in test:
                converter.experimental_new_quantizer = test['experimental_new_quantizer']

            tflite_model = converter.convert()

            output_path = OUTPUT_DIR / f"{test['name']}.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            # Count ops
            op_counts = count_ops(str(output_path))
            print(f"  Operations: {op_counts}")

            # Check for MINIMUM/RELU ops (unfused activations)
            has_minimum = op_counts.get('MINIMUM', 0) > 0
            has_relu = op_counts.get('RELU', 0) > 0
            print(f"  Has MINIMUM ops: {has_minimum}")
            print(f"  Has RELU ops: {has_relu}")

            # Test inference
            top5_idx, top5_conf = test_inference(str(output_path), TEST_IMAGE)
            top1_label = labels[top5_idx[0]+1] if top5_idx[0]+1 < len(labels) else f"idx_{top5_idx[0]}"
            print(f"  Top-1: {top1_label} ({top5_conf[0]:.4f})")

            results.append({
                'name': test['name'],
                'has_minimum': has_minimum,
                'has_relu': has_relu,
                'top1': top1_label,
                'conf': top5_conf[0]
            })

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Compare with ST
    st_ops = count_ops(str(MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite"))
    print(f"\nST model ops: {st_ops}")

    for r in results:
        status = "✓" if r['top1'] not in ['safe', 'screw', 'padlock', 'vault', 'safety pin'] else "✗"
        print(f"\n{r['name']}:")
        print(f"  MINIMUM: {r['has_minimum']}, RELU: {r['has_relu']}")
        print(f"  Top-1: {r['top1']} ({r['conf']:.4f}) {status}")

if __name__ == "__main__":
    main()
