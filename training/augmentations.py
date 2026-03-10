"""
Background-aware image augmentation for training.
Uses rembg (U2-Net) to generate foreground masks at load time,
then replaces background with random colors/noise/gradients during augmentation.
Foreground (board) pixels are always preserved.
"""

import numpy as np
import tensorflow as tf

MODEL_INPUT_SIZE = (224, 224)


def generate_mask(pil_img):
    """
    Generate a foreground mask for a PIL image using rembg.
    Returns mask as float32 numpy array [H, W, 1] in [0, 1].
    """
    from rembg import remove
    pil_rgba = remove(pil_img)
    mask = np.array(pil_rgba)[:, :, 3:4].astype(np.float32) / 255.0
    return mask


def augment_with_bg_replacement(img, mask):
    """
    Augment a training image with background replacement.
    The foreground (board) pixels are preserved; only the background is replaced.

    img: [224, 224, 3] float32 in [-1, 1]
    mask: [224, 224, 1] float32 in [0, 1], 1=foreground
    Returns: augmented [224, 224, 3] image
    """
    # --- Geometric augmentations applied to both image and mask ---

    # Horizontal flip
    do_flip = tf.random.uniform([]) > 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    # Random rotation ±30 degrees
    angle = tf.random.uniform([], -30.0, 30.0) * (np.pi / 180.0)
    img = _rotate_image(img, angle)
    mask = _rotate_image(mask, angle)

    # Random crop and resize (same crop for both)
    crop_fraction = tf.random.uniform([], 0.75, 1.0)
    crop_size = tf.cast(tf.cast(MODEL_INPUT_SIZE[0], tf.float32) * crop_fraction, tf.int32)
    combined = tf.concat([img, mask], axis=-1)  # [H, W, 4]
    combined = tf.image.random_crop(combined, [crop_size, crop_size, 4])
    combined = tf.image.resize(combined, MODEL_INPUT_SIZE)
    img = combined[:, :, :3]
    mask = combined[:, :, 3:]

    # Binarize mask after geometric transforms
    mask = tf.cast(mask > 0.5, tf.float32)

    # --- Background replacement (70% of the time) ---
    if tf.random.uniform([]) > 0.3:
        bg_type = tf.random.uniform([], 0.0, 1.0)
        if bg_type < 0.4:
            # Solid random color
            bg_color = tf.random.uniform([1, 1, 3], -1.0, 1.0)
            new_bg = tf.ones_like(img) * bg_color
        elif bg_type < 0.7:
            # Gaussian noise
            new_bg = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.5)
        else:
            # Smooth gradient
            y_grad = tf.linspace(0.0, 1.0, tf.shape(img)[0])
            y_grad = tf.reshape(y_grad, [-1, 1, 1])
            color_top = tf.random.uniform([1, 1, 3], -1.0, 1.0)
            color_bot = tf.random.uniform([1, 1, 3], -1.0, 1.0)
            new_bg = color_top * (1.0 - y_grad) + color_bot * y_grad

        img = img * mask + new_bg * (1.0 - mask)

    # --- Color augmentations on the whole composited image ---
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
    img = img + tf.random.uniform([1, 1, 3], -0.1, 0.1)

    # Random Gaussian blur
    if tf.random.uniform([]) > 0.5:
        img = _gaussian_blur(img)

    img = tf.clip_by_value(img, -1.0, 1.0)
    return img


def augment_simple(img):
    """
    Standard augmentation without mask (for replay images).
    """
    img = tf.image.random_flip_left_right(img)

    angle = tf.random.uniform([], -30.0, 30.0) * (np.pi / 180.0)
    img = _rotate_image(img, angle)

    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
    img = img + tf.random.uniform([1, 1, 3], -0.1, 0.1)

    crop_fraction = tf.random.uniform([], 0.75, 1.0)
    crop_size = tf.cast(tf.cast(MODEL_INPUT_SIZE[0], tf.float32) * crop_fraction, tf.int32)
    img = tf.image.random_crop(img, [crop_size, crop_size, 3])
    img = tf.image.resize(img, MODEL_INPUT_SIZE)

    if tf.random.uniform([]) > 0.5:
        img = _gaussian_blur(img)

    img = tf.clip_by_value(img, -1.0, 1.0)
    return img


# --- Internal helpers ---

def _rotate_image(image, angle):
    """Rotate image by angle (radians) using affine transform."""
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    cx, cy = MODEL_INPUT_SIZE[0] / 2.0, MODEL_INPUT_SIZE[1] / 2.0
    transform = [cos_a, sin_a, cx - cos_a * cx - sin_a * cy,
                 -sin_a, cos_a, cy + sin_a * cx - cos_a * cy,
                 0.0, 0.0]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.cast(tf.expand_dims(transform, 0), tf.float32),
        output_shape=MODEL_INPUT_SIZE,
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0
    )
    return tf.squeeze(image, 0)


def _gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to a single image."""
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    image = tf.expand_dims(image, 0)
    image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(image, 0)

