import os
import random
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append("..")
sys.path.append("../quantization")
from quantization.classes import IMAGENET2012_CLASSES

MODEL_INPUT_SIZE = (224, 224)

def to_class_index(synset_id, is_tfhub_model=False):
    base = list(IMAGENET2012_CLASSES.keys()).index(synset_id)
    return base + 1 if is_tfhub_model else base

def make_calibration_dataset(image_dir, num_images=500, is_tfhub_model=False):
    """Scan image_dir recursively and locate all images, then shuffle and pick 500."""
    file_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in file_extensions:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        raise RuntimeError("No images found!")

    random.seed(134)
    random.shuffle(image_paths)
    random.seed()

    image_paths = image_paths[:num_images]
    if len(image_paths) != num_images:
        raise RuntimeError(f"Expected {num_images}, found {len(image_paths)}")

    images = []
    classes = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB').resize(MODEL_INPUT_SIZE)
            if is_tfhub_model:
                img_array = np.array(img, dtype=np.float32) / 255.0
            else:
                img_array = np.array(img, dtype=np.float32)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            images.append(img_array)

            synset_id = os.path.basename(os.path.dirname(path))
            classes.append(to_class_index(synset_id, is_tfhub_model))

        except Exception as ex:
            raise RuntimeError(f"Failed to load {path}: {ex}")

    return np.stack(images, axis=0), np.array(classes, dtype=np.int32)

source = "./imagenet-val"
if len(sys.argv) > 1 and sys.argv[1] is not None:
    source = sys.argv[1]

print(f"Loading images from {source}...")
images_np_array, classes_np_array = make_calibration_dataset(source, num_images=500, is_tfhub_model=False)
print("Per-channel means:", [np.mean(images_np_array[..., i]) for i in range(3)])
print("Per-channel stds:", [np.std(images_np_array[..., i]) for i in range(3)])
print("Min:", images_np_array.min(), "Max", images_np_array.max())  # Should show ~-1.0, ~1.0

print("Saving to calibration.npz...")
np.savez_compressed("calibration.npz", representative_data=images_np_array, labels=classes_np_array)

print("Saving to calibration-stedgeai.npz (100 images - suitable for ST Edge Ai Web interface)...")
np.savez_compressed("calibration-stedgeai.npz", representative_data=images_np_array[:100])
