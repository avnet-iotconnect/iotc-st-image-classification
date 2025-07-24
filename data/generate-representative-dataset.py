import os
import random
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_INPUT_SIZE = (224, 224)


def make_calibration_dataset(image_dir, num_images=500):
    """Scan image_dir recursively and locate all images, then shuffle and pick 500 ."""
    file_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    # Find all images (case-insensitive)
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in file_extensions:
                image_paths.append(os.path.join(root, f))

    if not image_paths or len(image_paths) == 0:
        raise RuntimeError("No images found!")

    random.seed(134) # so we get the predictable set
    random.shuffle(image_paths)
    random.seed() # revert to true random

    image_paths = image_paths[:num_images]
    if len(image_paths) != num_images:
        raise RuntimeError(f"Expected {num_images}, but found only {len(image_paths)}!")


    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB').resize(MODEL_INPUT_SIZE)
            img_array = np.array(img, dtype=np.float32)
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            images.append(img_preprocessed)
        except Exception as ex:
            raise RuntimeError(f"Failed to load {path}! Error was", ex)

    if len(images) < num_images:
        print(f"WARNING: Expected {num_images}, but some failed to load {len(image_paths)}!")

    return np.stack(images, axis=0)

source="./imagenet-val"
if len(sys.argv) > 1 and sys.argv[1] is not None:
    source = sys.argv[1]

print(f"Loading images from {source}...")
images_np_array = make_calibration_dataset(source, num_images=500)

print("Saving to calibration.npz...")
np.savez_compressed("calibration.npz",  representative_data=images_np_array)

print("Saving to calibration-small.npz (100 images)...")
np.savez_compressed("calibration-small.npz",  representative_data=images_np_array[:100])
