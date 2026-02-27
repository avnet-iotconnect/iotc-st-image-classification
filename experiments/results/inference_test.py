#!/usr/bin/env python3
"""
Inference comparison: Run same image through all models and compare outputs
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
LABELS_FILE = MODELS_DIR / "ImageNetLabels.txt"

MODELS = {
    "ST per-tensor": MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite",
    "My per-tensor": MODELS_DIR / "quantized-pt.tflite",
    "My per-channel": MODELS_DIR / "quantized-pc.tflite",
}

TEST_IMAGES = [
    DATA_DIR / "water_bottle_ILSVRC2012_val_00025139.JPEG",  # water bottle
]

# Add some images from imagenet-val
IMAGENET_VAL = DATA_DIR / "imagenet-val"
if IMAGENET_VAL.exists():
    # Get a few diverse classes
    class_dirs = sorted([d for d in IMAGENET_VAL.iterdir() if d.is_dir()])[:5]
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.JPEG"))[:1]
        TEST_IMAGES.extend(images)

def load_labels():
    """Load ImageNet labels (1001 classes, index 0 = background)"""
    with open(LABELS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]

def run_inference(model_path, image_path):
    """Run inference on a single image"""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

    # Prepare input based on expected dtype
    in_dtype = in_details['dtype']
    if in_dtype == np.uint8:
        tensor = img_np.astype(np.uint8)
    elif in_dtype == np.int8:
        tensor = (img_np.astype(np.int32) - 128).astype(np.int8)
    else:
        # Float - use MobileNetV2 preprocessing
        tensor = (img_np.astype(np.float32) / 127.5) - 1.0

    tensor = np.expand_dims(tensor, 0)
    interp.set_tensor(in_details['index'], tensor)
    interp.invoke()

    output = interp.get_tensor(out_details['index'])[0]

    # Get top-5 predictions
    top5_idx = output.argsort()[-5:][::-1]
    top5_conf = output[top5_idx]

    # Normalize confidences if needed
    if out_details['dtype'] == np.uint8:
        top5_conf = top5_conf.astype(np.float32) / 255.0

    return top5_idx, top5_conf, output

def main():
    labels = load_labels()
    print(f"Loaded {len(labels)} labels")
    print(f"Testing {len(TEST_IMAGES)} images\n")

    for img_path in TEST_IMAGES:
        if not img_path.exists():
            continue

        print(f"\n{'='*80}")
        print(f"IMAGE: {img_path.name}")
        print(f"{'='*80}")

        results = {}

        for model_name, model_path in MODELS.items():
            if not model_path.exists():
                print(f"  {model_name}: SKIP (not found)")
                continue

            try:
                top5_idx, top5_conf, raw_output = run_inference(model_path, img_path)
                results[model_name] = (top5_idx, top5_conf, raw_output)

                print(f"\n  {model_name}:")
                print(f"    Output stats: min={raw_output.min():.4f} max={raw_output.max():.4f} std={raw_output.std():.4f}")
                for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
                    label = labels[idx] if idx < len(labels) else f"idx_{idx}"
                    marker = "<<<" if i == 0 else ""
                    print(f"    #{i+1}: {conf:7.4f} [{idx:4d}] {label} {marker}")
            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")

        # Compare if we have multiple results
        if len(results) >= 2:
            print(f"\n  COMPARISON:")
            model_names = list(results.keys())
            for i, name1 in enumerate(model_names):
                for name2 in model_names[i+1:]:
                    idx1 = results[name1][0][0]  # Top-1 class
                    idx2 = results[name2][0][0]
                    match = "✓ MATCH" if idx1 == idx2 else "✗ DIFFER"
                    print(f"    {name1} vs {name2}: {match} ({labels[idx1]} vs {labels[idx2]})")

if __name__ == "__main__":
    main()
