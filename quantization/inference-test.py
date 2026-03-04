#!/usr/bin/env python3
"""
Single-image inference for MobileNet-V2 on PC:

  * one function for the original Keras/TF model (expects float32 in [-1,1])
  * one function for a quantised TFLite model (accepts any runtime dtype)

Both return (top-1 index, confidence).
"""

import os
import sys

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_INPUT_SIZE = (224, 224)
DEFAULT_IMG = "../data/water_bottle_ILSVRC2012_val_00025139.JPEG"



# ------------------------------------------------------------------
# 1. Keras (standalone Keras 3) — use this, not tf.keras
# ------------------------------------------------------------------
def keras_inference(model_path=None, image_path=DEFAULT_IMG):
    model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet'
    )
    img = Image.open(image_path).convert("RGB").resize(MODEL_INPUT_SIZE)
    input_tensor = np.array(img, dtype=np.float32)

    # MobileNetV2 expects [-1, 1]
    input_tensor = (input_tensor / 127.5) - 1.0

    input_tensor = np.expand_dims(input_tensor, 0)  # Shape: [1, 224, 224, 3]
    preds = model.predict(input_tensor, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(preds[0, idx])
    return idx, conf


# ------------------------------------------------------------------
# 2. TFLite – input / output dtype agnostic
# ------------------------------------------------------------------
def tflite_inference(tflite_path, image_path=DEFAULT_IMG):
    """Works no matter whether the TFLite model expects
       uint8 / int8 / float16 / float32."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_details  = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    img = Image.open(image_path).convert("RGB").resize(MODEL_INPUT_SIZE)
    img_np = np.array(img)

    # --- build tensor with correct dtype / shape -------------------
    in_dtype = in_details['dtype']
    if in_dtype == np.uint8 or in_dtype == np.int8:
        # Quantized model: raw 0-255 pixels — the quantization params
        # bake in the [-1,1] mapping via scale/zero_point
        tensor = img_np.astype(in_dtype)
    elif in_dtype == np.float32 or in_dtype == np.float16:
        # Float model: MobileNetV2 expects [-1, 1]
        tensor = (img_np.astype(np.float32) / 127.5) - 1.0
    else:
        raise TypeError(f"Unsupported input dtype {in_dtype}")

    tensor = np.expand_dims(tensor, 0)
    interp.set_tensor(in_details['index'], tensor)
    interp.invoke()

    preds = interp.get_tensor(out_details['index'])
    idx = int(np.argmax(preds))
    conf = float(preds[0, idx])
    return idx, conf


# ------------------------------------------------------------------
# Quick CLI test
# ------------------------------------------------------------------
if __name__ == "__main__":
    img = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMG
    tflite = sys.argv[1] if len(sys.argv) > 1 else None

    import classes

    if tflite and os.path.isfile(tflite):
        print("TFLite model:")
        idx, conf = tflite_inference(tflite, img)
        print(f"  index={idx:4d}  confidence={conf:.4f}")
        print(f"{list(classes.IMAGENET2012_CLASSES.values())[idx]}")

    print("TF / Keras model:")
    idx, conf = keras_inference(image_path=img)
    print(f"  index={idx:4d}  confidence={conf:.4f}")
    print(f"{list(classes.IMAGENET2012_CLASSES.values())[idx]}")
