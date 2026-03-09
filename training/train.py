import os
import sys

try:
    import silence_tensorflow.auto
except ImportError: pass

import argparse
import random
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from quantization.classes import IMAGENET2012_CLASSES

MODEL_INPUT_SIZE = (224, 224)
NUM_ORIGINAL_CLASSES = 1000
SYNSET_IDS = list(IMAGENET2012_CLASSES.keys())
NUM_TOTAL_CLASSES = len(SYNSET_IDS)  # 1002
NEW_CLASS_SYNSETS = [s for s in SYNSET_IDS if s.startswith("n9999")]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}

def synset_to_index(synset_id):
    """Convert synset ID to 0-based class index (matching MobileNetV2 output order)."""
    return SYNSET_IDS.index(synset_id)

def load_image(path):
    """Load and preprocess a single image for MobileNetV2 ([-1, 1] range)."""
    img = Image.open(path).convert('RGB').resize(MODEL_INPUT_SIZE)
    img_array = np.array(img, dtype=np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def scan_image_dir(base_dir, synset_filter=None):
    """
    Scan base_dir/<synset_id>/ for images.
    Returns list of (image_path, class_index) tuples.
    synset_filter: optional set/list of synset IDs to include. None = all found.
    """
    samples = []
    if not os.path.isdir(base_dir):
        return samples
    for synset_id in sorted(os.listdir(base_dir)):
        synset_path = os.path.join(base_dir, synset_id)
        if not os.path.isdir(synset_path):
            continue
        if synset_id not in SYNSET_IDS:
            continue
        if synset_filter is not None and synset_id not in synset_filter:
            continue
        class_idx = synset_to_index(synset_id)
        for fname in sorted(os.listdir(synset_path)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                samples.append((os.path.join(synset_path, fname), class_idx))
    return samples

def load_samples(sample_list):
    """Load a list of (path, class_index) into numpy arrays."""
    images, labels = [], []
    for path, cls in sample_list:
        try:
            images.append(load_image(path))
            labels.append(cls)
        except Exception as e:
            print(f"  Warning: skipping {path}: {e}")
    return np.stack(images), np.array(labels, dtype=np.int32)


# --- Augmentation ---

def augment_image(img):
    """
    Heavy augmentation for a single image (expected shape [224,224,3], range [-1, 1]).
    Random rotation, brightness/contrast jitter, crop+resize, horizontal flip, color shift, blur.
    """
    img = tf.image.random_flip_left_right(img)

    # Random rotation ±30 degrees
    angle = tf.random.uniform([], -30.0, 30.0) * (np.pi / 180.0)
    img = rotate_image(img, angle)

    # Random brightness and contrast
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)

    # Slight color shift per channel
    img = img + tf.random.uniform([1, 1, 3], -0.1, 0.1)

    # Random crop and resize back (simulate scale jitter)
    crop_fraction = tf.random.uniform([], 0.75, 1.0)
    crop_size = tf.cast(tf.cast(MODEL_INPUT_SIZE[0], tf.float32) * crop_fraction, tf.int32)
    img = tf.image.random_crop(img, [crop_size, crop_size, 3])
    img = tf.image.resize(img, MODEL_INPUT_SIZE)

    # Random Gaussian blur (approximate with avg pool trick)
    if tf.random.uniform([]) > 0.5:
        img = gaussian_blur(img)

    img = tf.clip_by_value(img, -1.0, 1.0)
    return img

def rotate_image(image, angle):
    """Rotate image by angle (radians) using affine transform."""
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    # Center of image
    cx, cy = MODEL_INPUT_SIZE[0] / 2.0, MODEL_INPUT_SIZE[1] / 2.0
    # Inverse rotation matrix for pixel mapping
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

def gaussian_blur(image, kernel_size=5, sigma=1.0):
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


# --- Dataset construction ---

def make_augmented_dataset(images, labels, augment_fn, repeat_factor, batch_size):
    """
    Build a tf.data.Dataset that applies augmentation and repeats.
    repeat_factor: how many augmented copies per original image per epoch.
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.repeat(repeat_factor)
    ds = ds.shuffle(len(images) * repeat_factor)
    ds = ds.map(lambda x, y: (augment_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_plain_dataset(images, labels, batch_size):
    """Build a tf.data.Dataset without augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.shuffle(len(images))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# --- Model construction ---

def build_extended_model():
    """
    Build MobileNetV2 extended from 1000 to NUM_TOTAL_CLASSES outputs.
    Freeze all base layers. Copy original weights into the new head.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet'
    )

    # Extract the original classification head weights
    original_dense = base.get_layer('predictions')
    orig_weights, orig_biases = original_dense.get_weights()
    # orig_weights: (1280, 1000), orig_biases: (1000,)

    # Remove the original classification head
    feature_model = keras.Model(inputs=base.input, outputs=base.layers[-2].output, name='mobilenetv2_features')

    # Freeze all feature layers
    for layer in feature_model.layers:
        layer.trainable = False

    # Build extended head
    x = feature_model.output
    output = keras.layers.Dense(NUM_TOTAL_CLASSES, activation='softmax', name='predictions_extended')(x)
    model = keras.Model(inputs=feature_model.input, outputs=output)

    # Copy original 1000-class weights into the new head, zero-init new class weights
    new_dense = model.get_layer('predictions_extended')
    new_weights = np.zeros((orig_weights.shape[0], NUM_TOTAL_CLASSES), dtype=np.float32)
    new_biases = np.zeros(NUM_TOTAL_CLASSES, dtype=np.float32)
    new_weights[:, :NUM_ORIGINAL_CLASSES] = orig_weights
    new_biases[:NUM_ORIGINAL_CLASSES] = orig_biases
    # Initialize new class biases slightly negative so they don't fire on unrelated inputs
    new_biases[NUM_ORIGINAL_CLASSES:] = -2.0
    new_dense.set_weights([new_weights, new_biases])

    print(f"Model extended: {NUM_ORIGINAL_CLASSES} -> {NUM_TOTAL_CLASSES} classes")
    print(f"Trainable parameters: {sum(np.prod(v.shape) for v in model.trainable_weights):,}")
    print(f"Total parameters: {sum(np.prod(v.shape) for v in model.weights):,}")
    return model


# --- Training ---

def train(args):
    train_dir = os.path.join(args.train_data_dir, 'train')
    val_dir = os.path.join(args.train_data_dir, 'imagenet-val')

    # Load new-class training data
    print(f"\nScanning training data from {train_dir}...")
    new_class_train = scan_image_dir(train_dir, synset_filter=set(NEW_CLASS_SYNSETS))
    if not new_class_train:
        raise RuntimeError(f"No new-class training images found in {train_dir}. "
                           f"Expected folders: {NEW_CLASS_SYNSETS}")

    # Count per class and warn
    from collections import Counter
    train_counts = Counter(cls for _, cls in new_class_train)
    for synset in NEW_CLASS_SYNSETS:
        idx = synset_to_index(synset)
        count = train_counts.get(idx, 0)
        label = IMAGENET2012_CLASSES[synset]
        print(f"  {synset} ({label}): {count} training images")
        if count < 10:
            print(f"  *** WARNING: fewer than 10 images for class '{label}' — results may be poor ***")

    print(f"\nLoading new-class training images...")
    new_train_images, new_train_labels = load_samples(new_class_train)
    print(f"  Loaded {len(new_train_images)} new-class training images")

    # Load replay data: sample from existing ImageNet classes to stabilize old-class outputs
    print(f"\nLoading replay data from {val_dir} (sampling existing classes)...")
    all_existing = scan_image_dir(val_dir, synset_filter=set(SYNSET_IDS[:NUM_ORIGINAL_CLASSES]))
    random.seed(42)
    random.shuffle(all_existing)

    # Use up to replay_per_class images per existing class to keep training balanced
    replay_per_class = args.replay_per_class
    replay_counts = Counter()
    replay_samples = []
    for path, cls in all_existing:
        if replay_counts[cls] < replay_per_class:
            replay_samples.append((path, cls))
            replay_counts[cls] += 1

    print(f"  Sampling up to {replay_per_class} images per existing class...")
    replay_images, replay_labels = load_samples(replay_samples)
    print(f"  Loaded {len(replay_images)} replay images across {len(replay_counts)} classes")

    # Load validation data for new classes
    print(f"\nLoading validation data for new classes from {val_dir}...")
    new_class_val = scan_image_dir(val_dir, synset_filter=set(NEW_CLASS_SYNSETS))
    val_counts = Counter(cls for _, cls in new_class_val)
    for synset in NEW_CLASS_SYNSETS:
        idx = synset_to_index(synset)
        count = val_counts.get(idx, 0)
        label = IMAGENET2012_CLASSES[synset]
        print(f"  {synset} ({label}): {count} validation images")
        if count < 5:
            print(f"  *** WARNING: fewer than 5 validation images for class '{label}' ***")

    if new_class_val:
        val_images, val_labels = load_samples(new_class_val)
        print(f"  Loaded {len(val_images)} new-class validation images")
    else:
        print("  No new-class validation images found. Skipping new-class validation.")
        val_images, val_labels = None, None

    # Build model
    print("\nBuilding extended MobileNetV2 model...")
    model = build_extended_model()

    # Build training datasets
    # New-class data: heavy augmentation, repeated ~15x to multiply small dataset
    augment_repeat = args.augment_repeat
    print(f"\nNew-class augmentation repeat factor: {augment_repeat}x")
    new_ds = make_augmented_dataset(new_train_images, new_train_labels, augment_image, augment_repeat, args.batch_size)

    # Replay data: no augmentation (these are just anchors for old-class stability)
    replay_ds = make_plain_dataset(replay_images, replay_labels, args.batch_size)

    # Interleave new-class and replay datasets
    # Weight toward new classes since that's what we're learning
    train_ds = tf.data.Dataset.sample_from_datasets(
        [new_ds, replay_ds],
        weights=[0.6, 0.4],
        stop_on_empty_dataset=False
    )

    # Compute class weights to up-weight new classes further
    total_new = len(new_train_images) * augment_repeat
    total_replay = len(replay_images)
    total = total_new + total_replay
    # New classes get higher weight, replay classes get lower weight
    class_weight = {}
    for i in range(NUM_ORIGINAL_CLASSES):
        class_weight[i] = 0.5
    for synset in NEW_CLASS_SYNSETS:
        idx = synset_to_index(synset)
        class_weight[idx] = total / (2.0 * train_counts.get(idx, 1) * augment_repeat)

    new_class_weights_str = {IMAGENET2012_CLASSES[s]: f"{class_weight[synset_to_index(s)]:.2f}" for s in NEW_CLASS_SYNSETS}
    print(f"Class weights for new classes: {new_class_weights_str}")
    print(f"Class weight for replay classes: 0.50")

    # Steps per epoch: enough to see all augmented new-class images once
    steps_per_epoch = max(total_new // args.batch_size, total_replay // args.batch_size, 50)
    print(f"Steps per epoch: {steps_per_epoch}")

    # Compile and train
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nTraining for {args.epochs} epochs (lr={args.learning_rate})...\n")
    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight,
        verbose=1
    )

    # --- Post-training validation ---
    print("\n" + "=" * 60)
    print("POST-TRAINING VALIDATION")
    print("=" * 60)

    # Validate new classes
    if val_images is not None:
        print("\nNew-class validation:")
        new_preds = model.predict(val_images, verbose=0)
        new_pred_classes = np.argmax(new_preds, axis=1)
        for synset in NEW_CLASS_SYNSETS:
            idx = synset_to_index(synset)
            mask = val_labels == idx
            if mask.sum() == 0:
                continue
            correct = (new_pred_classes[mask] == idx).sum()
            total_cls = mask.sum()
            label = IMAGENET2012_CLASSES[synset]
            print(f"  {label}: {correct}/{total_cls} correct ({100 * correct / total_cls:.1f}%)")

            # Show what misclassified images were predicted as
            wrong_mask = mask & (new_pred_classes != idx)
            if wrong_mask.sum() > 0:
                wrong_preds = new_pred_classes[wrong_mask]
                for wp in wrong_preds:
                    wrong_label = IMAGENET2012_CLASSES[SYNSET_IDS[wp]]
                    print(f"    misclassified as: {wrong_label} (index {wp})")

        overall_acc = (new_pred_classes == val_labels).mean()
        print(f"  Overall new-class accuracy: {overall_acc * 100:.1f}%")

    # Quick spot-check on a few existing classes
    print("\nExisting-class spot check (5 random classes, up to 10 images each):")
    random.seed(123)
    spot_check_synsets = random.sample(SYNSET_IDS[:NUM_ORIGINAL_CLASSES], min(5, NUM_ORIGINAL_CLASSES))
    for synset in spot_check_synsets:
        idx = synset_to_index(synset)
        synset_dir = os.path.join(val_dir, synset)
        if not os.path.isdir(synset_dir):
            continue
        samples = scan_image_dir(val_dir, synset_filter={synset})[:10]
        if not samples:
            continue
        imgs, lbls = load_samples(samples)
        preds = np.argmax(model.predict(imgs, verbose=0), axis=1)
        correct = (preds == lbls).sum()
        label = IMAGENET2012_CLASSES[synset]
        print(f"  {label}: {correct}/{len(lbls)} ({100 * correct / len(lbls):.1f}%)")

    # Save model
    out_path = os.path.join(args.model_dir, args.output_model)
    print(f"\nSaving model to {out_path}...")
    model.save(out_path)
    print(f"Model saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 with new classes")

    # SageMaker-compatible arguments
    parser.add_argument('--train-data-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING') or '../data',
                        help="Root data directory containing train/ and imagenet-val/ subdirs. Default: ../data")
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR') or '../models',
                        help="Directory to save the fine-tuned model. Default: ../models")
    parser.add_argument('--output-model', type=str, default='mobilenetv2-finetuned.keras',
                        help="Output model filename. Default: mobilenetv2-finetuned.keras")

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=8,
                        help="Number of training epochs. Default: 8")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Training batch size. Default: 16")
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help="Learning rate for Adam optimizer. Default: 1e-4")
    parser.add_argument('--augment-repeat', type=int, default=15,
                        help="Augmentation repeat factor for new-class images. Default: 15")
    parser.add_argument('--replay-per-class', type=int, default=5,
                        help="Max images per existing ImageNet class for replay. Default: 5")

    args, _ = parser.parse_known_args()

    print(f"Data root:    {args.train_data_dir}")
    print(f"Model dir:    {args.model_dir}")
    print(f"Output model: {args.output_model}")
    print(f"New classes:  {NEW_CLASS_SYNSETS}")

    train(args)
    print("\nDone.")


if __name__ == '__main__':
    main()

