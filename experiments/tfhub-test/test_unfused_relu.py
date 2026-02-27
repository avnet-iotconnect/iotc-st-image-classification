#!/usr/bin/env python3
"""
Try to create a model with unfused activations like ST's model.
ST's model has MINIMUM and RELU ops separate from CONV_2D.

Approach: Build a custom model with explicit tf.clip_by_value for ReLU6
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

def test_inference(model_path, image_path, labels):
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

    print("Top-5 predictions:")
    for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
        label = labels[idx+1] if idx+1 < len(labels) else f"idx_{idx}"
        marker = " <<<" if i == 0 else ""
        print(f"  #{i+1}: {conf:.4f} [{idx:4d}] {label}{marker}")

    return top5_idx, top5_conf

def count_ops(model_path):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    from collections import Counter
    return dict(Counter(op.get('op_name', 'unknown') for op in ops))

def main():
    import sys
    # Write to file for debugging
    log_file = open('/tmp/unfused_debug.txt', 'w')
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("="*70)
    log("Testing model with explicit clip_by_value (unfused ReLU6)")
    log("="*70)

    # Load labels
    labels_path = MODELS_DIR / "ImageNetLabels.txt"
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    cal_images = create_calibration_data()
    log(f"Calibration images: {cal_images.shape}")

    # Load base model
    log("\nLoading MobileNetV2...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=True
    )

    # Convert to SavedModel format first (sometimes helps with op fusion control)
    saved_model_dir = OUTPUT_DIR / "saved_model_temp"
    log(f"\nSaving to SavedModel format: {saved_model_dir}")
    tf.saved_model.save(base_model, str(saved_model_dir))

    # Try converting from SavedModel with different settings
    log("\nConverting from SavedModel...")

    def representative_dataset():
        for i in range(len(cal_images)):
            yield [cal_images[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter._experimental_disable_per_channel = True

    # Try to disable op fusion
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    output_path = OUTPUT_DIR / "from_savedmodel_pt.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    log(f"Saved: {output_path}")

    ops = count_ops(str(output_path))
    log(f"\nOperations: {ops}")
    log(f"Has MINIMUM: {ops.get('MINIMUM', 0)}")
    log(f"Has RELU: {ops.get('RELU', 0)}")

    log("\nInference test:")
    top5_idx, top5_conf = test_inference(str(output_path), TEST_IMAGE, labels)
    for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
        label = labels[idx+1] if idx+1 < len(labels) else f"idx_{idx}"
        log(f"  #{i+1}: {conf:.4f} [{idx:4d}] {label}")

    # Compare with ST model
    log("\n" + "="*70)
    log("ST model for reference:")
    st_ops = count_ops(str(MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite"))
    log(f"ST ops: {st_ops}")

    # Cleanup
    import shutil
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    log_file.close()

if __name__ == "__main__":
    main()
