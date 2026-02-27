#!/usr/bin/env python3
"""
TFHub MobileNetV2 Quantization Test
Test if TFHub's BN-folded MobileNetV2 works better for per-tensor quantization
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

# Test image
TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"

def create_tfhub_model():
    """Create MobileNetV2 from TFHub with BN already folded"""
    print("Creating TFHub MobileNetV2 model...")

    # TFHub feature vector model (has BN folded)
    # This is a 1280-dim feature extractor without classification head
    hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"

    inputs = tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32)

    # TFHub expects input in [0, 1] range
    # The hub layer handles the internal preprocessing
    hub_layer = hub.KerasLayer(hub_url, trainable=False)
    features = hub_layer(inputs)

    # Add classification head (1001 classes for ImageNet with background)
    outputs = tf.keras.layers.Dense(1001, activation='softmax', name='predictions')(features)

    model = tf.keras.Model(inputs, outputs)
    return model

def create_keras_model():
    """Create standard Keras MobileNetV2 for comparison"""
    print("Creating Keras MobileNetV2 model...")
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=True
    )
    return model

def create_calibration_data():
    """Create calibration dataset - use images from imagenet-val"""
    print("Creating calibration dataset...")

    imagenet_val = DATA_DIR / "imagenet-val"
    images = []

    # Collect images from multiple classes
    for class_dir in sorted(imagenet_val.iterdir())[:50]:  # 50 classes
        if not class_dir.is_dir():
            continue
        class_images = list(class_dir.glob("*.JPEG"))[:2]  # 2 per class = 100 total
        for img_path in class_images:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            img_np = np.array(img, dtype=np.float32)
            images.append(img_np)

    images = np.stack(images, axis=0)
    print(f"  Collected {len(images)} calibration images")
    return images

def quantize_model(model, calibration_images, output_name, use_tfhub_preprocessing=False):
    """Quantize model to per-tensor int8"""
    print(f"\nQuantizing to {output_name}...")

    def representative_dataset():
        for i in range(len(calibration_images)):
            img = calibration_images[i:i+1]
            if use_tfhub_preprocessing:
                # TFHub expects [0, 1]
                img = img / 255.0
            else:
                # Keras expects [-1, 1]
                img = (img / 127.5) - 1.0
            yield [img.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    # Force per-tensor quantization
    converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()

    output_path = OUTPUT_DIR / output_name
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"  Saved to {output_path}")
    return output_path

def test_inference(model_path, image_path, use_tfhub_preprocessing=False):
    """Run inference and return top-5"""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

    # Input is uint8 0-255
    tensor = img_np.astype(np.uint8)
    tensor = np.expand_dims(tensor, 0)

    interp.set_tensor(in_details['index'], tensor)
    interp.invoke()

    output = interp.get_tensor(out_details['index'])[0]
    top5_idx = output.argsort()[-5:][::-1]
    top5_conf = output[top5_idx]

    return top5_idx, top5_conf

def main():
    print("="*70)
    print("TFHub MobileNetV2 Per-Tensor Quantization Test")
    print("="*70)

    # Load labels
    labels_path = MODELS_DIR / "ImageNetLabels.txt"
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    # Create calibration data
    calibration_images = create_calibration_data()

    # Test 1: TFHub model
    print("\n" + "="*70)
    print("TEST 1: TFHub MobileNetV2 (BN-folded)")
    print("="*70)

    try:
        tfhub_model = create_tfhub_model()
        tfhub_model.summary()

        tfhub_pt_path = quantize_model(
            tfhub_model,
            calibration_images,
            "tfhub_mobilenet_v2_pt.tflite",
            use_tfhub_preprocessing=True
        )

        print("\nInference test on water bottle image:")
        top5_idx, top5_conf = test_inference(tfhub_pt_path, TEST_IMAGE, use_tfhub_preprocessing=True)
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            label = labels[idx] if idx < len(labels) else f"idx_{idx}"
            print(f"  #{i+1}: {conf:.4f} [{idx:4d}] {label}")
    except Exception as e:
        print(f"ERROR: {e}")

    # Test 2: Standard Keras for comparison
    print("\n" + "="*70)
    print("TEST 2: Keras MobileNetV2 (standard)")
    print("="*70)

    try:
        keras_model = create_keras_model()

        keras_pt_path = quantize_model(
            keras_model,
            calibration_images,
            "keras_mobilenet_v2_pt.tflite",
            use_tfhub_preprocessing=False
        )

        print("\nInference test on water bottle image:")
        top5_idx, top5_conf = test_inference(keras_pt_path, TEST_IMAGE, use_tfhub_preprocessing=False)
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            # Keras uses 1000 classes (no background)
            label = labels[idx+1] if idx+1 < len(labels) else f"idx_{idx}"
            print(f"  #{i+1}: {conf:.4f} [{idx:4d}] {label}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
