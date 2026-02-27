#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

test_img = np.ones((224, 224, 3), dtype=np.float32) * 128.0

mv2_prep = tf.keras.applications.mobilenet_v2.preprocess_input(test_img.copy())
effv2_prep = tf.keras.applications.efficientnet_v2.preprocess_input(test_img.copy())

with open('/tmp/preprocess_check.txt', 'w') as f:
    f.write(f'MobileNetV2: range {mv2_prep.min():.3f} to {mv2_prep.max():.3f}\n')
    f.write(f'EfficientNetV2: range {effv2_prep.min():.3f} to {effv2_prep.max():.3f}\n')

    # Also check with 0 and 255
    test_0 = np.zeros((224, 224, 3), dtype=np.float32)
    test_255 = np.ones((224, 224, 3), dtype=np.float32) * 255.0

    f.write(f'\nMobileNetV2 with 0: {tf.keras.applications.mobilenet_v2.preprocess_input(test_0.copy())[0,0,0]:.3f}\n')
    f.write(f'MobileNetV2 with 255: {tf.keras.applications.mobilenet_v2.preprocess_input(test_255.copy())[0,0,0]:.3f}\n')
    f.write(f'EfficientNetV2 with 0: {tf.keras.applications.efficientnet_v2.preprocess_input(test_0.copy())[0,0,0]:.3f}\n')
    f.write(f'EfficientNetV2 with 255: {tf.keras.applications.efficientnet_v2.preprocess_input(test_255.copy())[0,0,0]:.3f}\n')

print("Done - check /tmp/preprocess_check.txt")
