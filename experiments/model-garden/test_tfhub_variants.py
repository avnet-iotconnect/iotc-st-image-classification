#!/usr/bin/env python3
"""
Search for pre-quantized MobileNetV2 models that might have unfused operations.
Try various sources: TFHub, Kaggle Models, TF Lite Model Zoo
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub
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

def main():
    log_path = OUTPUT_DIR / "tfhub_models_test.txt"

    with open(log_path, 'w') as log:
        def write(msg):
            print(msg)
            log.write(msg + '\n')
            log.flush()

        write("="*70)
        write("Testing TFHub Models for Unfused Operations")
        write("="*70)

        labels_path = MODELS_DIR / "ImageNetLabels.txt"
        with open(labels_path) as f:
            labels = [line.strip() for line in f.readlines()]

        cal_images = create_calibration_data()
        write(f"Calibration images: {cal_images.shape}")

        # Different TFHub MobileNetV2 variants
        tfhub_models = [
            # Classification models (have trained classification head)
            ("tfhub_mnv2_classification_v4", "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"),
            ("tfhub_mnv2_classification_v5", "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"),
            # Kaggle hosted
            ("kaggle_mnv2_classification", "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/100-224-classification/2"),
        ]

        for name, url in tfhub_models:
            write(f"\n{'='*70}")
            write(f"Model: {name}")
            write(f"URL: {url}")
            write(f"{'='*70}")

            try:
                # Load from TFHub
                write("  Loading model...")
                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                    hub.KerasLayer(url)
                ])

                # Check model summary for any hints
                write(f"  Output shape: {model.output_shape}")

                # Convert to TFLite with per-tensor (to see if structure differs)
                write("  Converting to TFLite (per-tensor)...")

                # Use appropriate preprocessing based on model
                if "preview" in url or "kaggle" in url.lower():
                    # These expect [0, 1] input
                    def representative_dataset():
                        for i in range(min(100, len(cal_images))):
                            yield [(cal_images[i:i+1] / 255.0).astype(np.float32)]
                else:
                    # Standard expects [-1, 1]
                    def representative_dataset():
                        for i in range(min(100, len(cal_images))):
                            yield [((cal_images[i:i+1] / 127.5) - 1.0).astype(np.float32)]

                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.float32
                converter._experimental_disable_per_channel = True

                tflite_model = converter.convert()

                output_path = OUTPUT_DIR / f"{name}_pt.tflite"
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)

                # Analyze ops
                ops = count_ops(str(output_path))
                write(f"\n  Operations: {ops}")
                write(f"  Has MINIMUM: {ops.get('MINIMUM', 0)}")
                write(f"  Has RELU: {ops.get('RELU', 0)}")
                write(f"  Has RELU6: {ops.get('RELU6', 0)}")

                is_unfused = ops.get('MINIMUM', 0) > 0 or ops.get('RELU', 0) > 0 or ops.get('RELU6', 0) > 0
                write(f"  Unfused activations: {'YES ✓' if is_unfused else 'NO ✗'}")

                # Test inference
                write(f"\n  Inference test:")
                top5_idx, top5_conf = test_inference(str(output_path), TEST_IMAGE, labels)
                for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
                    label = labels[idx] if idx < len(labels) else f"idx_{idx}"
                    marker = " <<<" if i == 0 else ""
                    write(f"    #{i+1}: {conf:.4f} [{idx:4d}] {label}{marker}")

                # Check if result is reasonable
                good_results = ['water bottle', 'pop bottle', 'bottle', 'water jug']
                top1_label = labels[top5_idx[0]] if top5_idx[0] < len(labels) else ""
                is_correct = any(g in top1_label.lower() for g in ['bottle', 'water', 'jug'])
                write(f"\n  Correct result: {'YES ✓' if is_correct else 'NO ✗'}")

            except Exception as e:
                write(f"  ERROR: {e}")

        write(f"\n\nResults written to: {log_path}")

if __name__ == "__main__":
    main()
