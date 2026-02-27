#!/usr/bin/env python3
"""
Quick test: Try MobileNetV3 and EfficientNetLite - these are newer models
designed with quantization and edge deployment in mind.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"

def count_ops(model_path):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    return dict(Counter(op.get('op_name', 'unknown') for op in ops))

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

def test_inference(model_path, image_path, labels):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

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

    if out_details['dtype'] == np.uint8:
        top5_conf = top5_conf.astype(np.float32) / 255.0

    return top5_idx, top5_conf

def test_model(model_name, model_fn, preprocess_fn, cal_images, labels, log):
    def write(msg):
        print(msg)
        log.write(msg + '\n')
        log.flush()

    write(f"\n{'='*70}")
    write(f"Model: {model_name}")
    write(f"{'='*70}")

    try:
        write("  Loading model...")
        model = model_fn(weights='imagenet', include_top=True)
        write(f"  Input shape: {model.input_shape}")
        write(f"  Output shape: {model.output_shape}")

        # Preprocess calibration data
        def representative_dataset():
            for i in range(min(100, len(cal_images))):
                img = cal_images[i:i+1]
                # Re-scale from [-1,1] back to [0,255] then apply model's preprocess
                img_uint8 = ((img + 1.0) * 127.5).astype(np.float32)
                img_prep = preprocess_fn(img_uint8)
                yield [img_prep]

        # Convert to per-tensor int8
        write("  Converting to TFLite (per-tensor)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        converter._experimental_disable_per_channel = True

        tflite_model = converter.convert()

        output_path = OUTPUT_DIR / f"{model_name}_pt.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # Analyze
        ops = count_ops(str(output_path))
        write(f"\n  Operations: {ops}")
        write(f"  Has MINIMUM: {ops.get('MINIMUM', 0)}")
        write(f"  Has RELU: {ops.get('RELU', 0)}")
        write(f"  Has HARD_SWISH: {ops.get('HARD_SWISH', 0)}")

        # Test inference
        write(f"\n  Inference test:")
        top5_idx, top5_conf = test_inference(str(output_path), TEST_IMAGE, labels)
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            label = labels[idx+1] if idx+1 < len(labels) else f"idx_{idx}"
            marker = " <<<" if i == 0 else ""
            write(f"    #{i+1}: {conf:.4f} [{idx:4d}] {label}{marker}")

        # Check correctness
        top1_label = labels[top5_idx[0]+1] if top5_idx[0]+1 < len(labels) else ""
        is_correct = any(g in top1_label.lower() for g in ['bottle', 'water', 'jug'])
        write(f"\n  Correct result: {'YES ✓' if is_correct else 'NO ✗'}")

        return is_correct

    except Exception as e:
        write(f"  ERROR: {e}")
        import traceback
        write(f"  {traceback.format_exc()}")
        return False

def main():
    log_path = OUTPUT_DIR / "newer_models_test.txt"

    with open(log_path, 'w') as log:
        def write(msg):
            print(msg)
            log.write(msg + '\n')
            log.flush()

        write("="*70)
        write("Testing Newer Models for Per-Tensor Quantization")
        write("="*70)
        write("These models are designed with edge deployment in mind:")
        write("- MobileNetV3: Successor to V2, uses hard-swish activation")
        write("- EfficientNetV2: Modern efficient architecture")
        write("="*70)

        labels_path = MODELS_DIR / "ImageNetLabels.txt"
        with open(labels_path) as f:
            labels = [line.strip() for line in f.readlines()]

        cal_images = create_calibration_data()
        write(f"\nCalibration images: {cal_images.shape}")

        # Models to test
        models_to_test = [
            ("MobileNetV3Small", tf.keras.applications.MobileNetV3Small,
             tf.keras.applications.mobilenet_v3.preprocess_input),
            ("MobileNetV3Large", tf.keras.applications.MobileNetV3Large,
             tf.keras.applications.mobilenet_v3.preprocess_input),
            ("EfficientNetV2B0", tf.keras.applications.EfficientNetV2B0,
             tf.keras.applications.efficientnet_v2.preprocess_input),
        ]

        results = []
        for name, model_fn, preprocess_fn in models_to_test:
            success = test_model(name, model_fn, preprocess_fn, cal_images, labels, log)
            results.append((name, success))

        write(f"\n{'='*70}")
        write("SUMMARY")
        write(f"{'='*70}")
        for name, success in results:
            status = "✓ WORKS" if success else "✗ FAILS"
            write(f"  {name}: {status}")

        write(f"\nResults written to: {log_path}")

if __name__ == "__main__":
    main()
