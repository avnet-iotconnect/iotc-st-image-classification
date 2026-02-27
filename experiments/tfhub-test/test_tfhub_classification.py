#!/usr/bin/env python3
"""
TFHub Classification Model Test
Use the full classification model from TFHub, not just feature vector
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from pathlib import Path
from PIL import Image

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"

def create_calibration_data():
    """Create calibration dataset"""
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

def test_inference(model_path, image_path):
    """Run inference"""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)
    tensor = img_np.astype(np.uint8)
    tensor = np.expand_dims(tensor, 0)

    interp.set_tensor(in_details['index'], tensor)
    interp.invoke()

    output = interp.get_tensor(out_details['index'])[0]
    top5_idx = output.argsort()[-5:][::-1]
    top5_conf = output[top5_idx]

    return top5_idx, top5_conf

def main():
    # Load labels
    labels_path = MODELS_DIR / "ImageNetLabels.txt"
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    calibration_images = create_calibration_data()
    print(f"Calibration images: {calibration_images.shape}")

    # TFHub classification model URL - expects [0, 1] input, outputs 1001 classes
    # This is the full trained model with classification head
    hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"

    print(f"\nLoading TFHub classification model from: {hub_url}")

    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        hub.KerasLayer(hub_url)
    ])

    # Test float model first
    print("\nTesting float model inference...")
    img = Image.open(TEST_IMAGE).convert("RGB").resize((224, 224))
    img_np = np.array(img, dtype=np.float32) / 255.0  # TFHub expects [0, 1]
    img_np = np.expand_dims(img_np, 0)

    preds = model.predict(img_np, verbose=0)
    top5_idx = preds[0].argsort()[-5:][::-1]
    top5_conf = preds[0][top5_idx]

    print("Float model results:")
    for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
        label = labels[idx] if idx < len(labels) else f"idx_{idx}"
        print(f"  #{i+1}: {conf:.4f} [{idx:4d}] {label}")

    # Now quantize with per-tensor
    print("\nQuantizing to per-tensor int8...")

    def representative_dataset():
        for i in range(len(calibration_images)):
            img = calibration_images[i:i+1]
            img = img / 255.0  # TFHub expects [0, 1]
            yield [img.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()

    output_path = OUTPUT_DIR / "tfhub_classification_pt.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved: {output_path}")

    # Test quantized model
    print("\nTesting quantized model inference...")
    top5_idx, top5_conf = test_inference(output_path, TEST_IMAGE)

    print("Quantized per-tensor results:")
    for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
        label = labels[idx] if idx < len(labels) else f"idx_{idx}"
        print(f"  #{i+1}: {conf:.4f} [{idx:4d}] {label}")

if __name__ == "__main__":
    main()
