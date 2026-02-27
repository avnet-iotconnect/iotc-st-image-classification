#!/usr/bin/env python3
"""
Test Quantization-Aware Training (QAT) for MobileNetV2
This adds fake quantization nodes during training, similar to what ST's model has.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# Try to import tfmot
try:
    import tensorflow_model_optimization as tfmot
    print(f"TFMOT version: {tfmot.__version__}")
except ImportError:
    print("ERROR: tensorflow-model-optimization not installed")
    print("Run: pip install tensorflow-model-optimization")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
TEST_IMAGE = DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG"

def create_calibration_data():
    """Create calibration/training dataset"""
    imagenet_val = DATA_DIR / "imagenet-val"
    images = []
    labels = []

    class_idx = 0
    for class_dir in sorted(imagenet_val.iterdir())[:100]:
        if not class_dir.is_dir():
            continue
        class_images = list(class_dir.glob("*.JPEG"))[:5]
        for img_path in class_images:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            img_np = np.array(img, dtype=np.float32)
            # Keras preprocessing: [-1, 1]
            img_np = (img_np / 127.5) - 1.0
            images.append(img_np)
            labels.append(class_idx)
        class_idx += 1

    return np.stack(images, axis=0), np.array(labels)

def test_inference(model_path, image_path):
    """Run inference on quantized model"""
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

def main():
    print("="*70)
    print("Quantization-Aware Training (QAT) Test")
    print("="*70)

    # Load labels
    labels_path = MODELS_DIR / "ImageNetLabels.txt"
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    # Create base model
    print("\n1. Creating base MobileNetV2 model...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=True
    )

    # Apply QAT
    print("\n2. Applying Quantization-Aware Training annotations...")

    # Use default quantization config - this adds FakeQuant nodes
    quantize_model = tfmot.quantization.keras.quantize_model

    # Clone and apply QAT
    qat_model = quantize_model(base_model)

    # Compile (required even for just running calibration)
    qat_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"   QAT model has {len(qat_model.layers)} layers")

    # Create calibration data
    print("\n3. Creating calibration dataset...")
    cal_images, cal_labels = create_calibration_data()
    print(f"   Calibration images: {cal_images.shape}")

    # Run a few training steps to calibrate the FakeQuant ranges
    print("\n4. Running calibration (few training steps)...")
    qat_model.fit(
        cal_images,
        cal_labels,
        epochs=1,
        batch_size=32,
        verbose=1
    )

    # Convert to TFLite with per-tensor quantization
    print("\n5. Converting to TFLite (per-tensor)...")

    def representative_dataset():
        for i in range(min(100, len(cal_images))):
            yield [cal_images[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()

    output_path = OUTPUT_DIR / "qat_mobilenet_v2_pt.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"   Saved: {output_path}")

    # Test inference
    print("\n6. Testing inference on water bottle image...")
    top5_idx, top5_conf = test_inference(output_path, TEST_IMAGE)

    print("\nQAT Per-tensor results:")
    for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
        # Keras uses 1000 classes, labels has 1001 (with background at 0)
        label = labels[idx+1] if idx+1 < len(labels) else f"idx_{idx}"
        marker = " <<<" if i == 0 else ""
        print(f"  #{i+1}: {conf:.4f} [{idx:4d}] {label}{marker}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
