"""
Fine-tune MobileNetV2 with 2 new classes (1000 -> 1002).
Simple, proven approach:
  1. Extend the classification head from 1000 to 1002 outputs
  2. Freeze all layers except the final Dense
  3. Build a single combined dataset: augmented new-class images + replay from ImageNet
  4. Train with standard categorical cross-entropy
  5. Validate on held-out new-class images + spot-check existing classes
"""

import os
import sys

try:
    import silence_tensorflow.auto
except ImportError:
    pass

import argparse
import random
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from quantization.classes import IMAGENET2012_CLASSES
from augmentations import augment_with_bg_replacement, augment_simple

MODEL_INPUT_SIZE = (224, 224)
NUM_ORIGINAL_CLASSES = 1000
SYNSET_IDS = list(IMAGENET2012_CLASSES.keys())
NUM_TOTAL_CLASSES = len(SYNSET_IDS)  # 1002
NEW_CLASS_SYNSETS = [s for s in SYNSET_IDS if s.startswith("n9999")]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}


def synset_to_index(synset_id):
    return SYNSET_IDS.index(synset_id)


def load_image(path):
    img = Image.open(path).convert('RGB').resize(MODEL_INPUT_SIZE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype=np.float32)
    )


def scan_image_dir(base_dir, synset_filter=None):
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
    images, labels = [], []
    for path, cls in sample_list:
        try:
            images.append(load_image(path))
            labels.append(cls)
        except Exception as e:
            print(f"  Warning: skipping {path}: {e}")
    return np.stack(images), np.array(labels, dtype=np.int32)


def load_samples_with_masks(sample_list):
    from rembg import remove
    images, masks, labels = [], [], []
    for i, (path, cls) in enumerate(sample_list):
        try:
            pil_img = Image.open(path).convert('RGB').resize(MODEL_INPUT_SIZE)
            pil_rgba = remove(pil_img)
            mask = np.array(pil_rgba)[:, :, 3:4].astype(np.float32) / 255.0

            img_array = np.array(pil_img, dtype=np.float32)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            images.append(img_array)
            masks.append(mask)
            labels.append(cls)
            print(f"    [{i + 1}/{len(sample_list)}] {os.path.basename(path)}: "
                  f"fg={100 * (mask > 0.5).mean():.0f}%", flush=True)
        except Exception as e:
            print(f"  Warning: skipping {path}: {e}")
    return np.stack(images), np.stack(masks), np.array(labels, dtype=np.int32)


# --- Model ---

def build_extended_model():
    """
    Load MobileNetV2 with ImageNet weights, extend head to 1002 classes.
    Only the new Dense layer is trainable.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=True, weights='imagenet'
    )
    original_dense = base.get_layer('predictions')
    orig_w, orig_b = original_dense.get_weights()

    feature_model = keras.Model(
        inputs=base.input,
        outputs=base.layers[-2].output,
        name='mobilenetv2_features'
    )
    for layer in feature_model.layers:
        layer.trainable = False

    x = feature_model.output
    output = keras.layers.Dense(
        NUM_TOTAL_CLASSES, activation='softmax', name='predictions_ext'
    )(x)
    model = keras.Model(inputs=feature_model.input, outputs=output)

    # Transfer original weights; init new classes with small random weights
    new_dense = model.get_layer('predictions_ext')
    new_w = np.zeros((orig_w.shape[0], NUM_TOTAL_CLASSES), dtype=np.float32)
    new_b = np.zeros(NUM_TOTAL_CLASSES, dtype=np.float32)
    new_w[:, :NUM_ORIGINAL_CLASSES] = orig_w
    new_b[:NUM_ORIGINAL_CLASSES] = orig_b
    # Small random init for new classes (not zero, not biased)
    rng = np.random.default_rng(42)
    new_w[:, NUM_ORIGINAL_CLASSES:] = rng.normal(0, 0.01, (orig_w.shape[0], NUM_TOTAL_CLASSES - NUM_ORIGINAL_CLASSES))
    new_b[NUM_ORIGINAL_CLASSES:] = 0.0
    new_dense.set_weights([new_w, new_b])

    trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    total = sum(np.prod(v.shape) for v in model.weights)
    print(f"Model: {NUM_ORIGINAL_CLASSES} -> {NUM_TOTAL_CLASSES} classes")
    print(f"Trainable: {trainable:,} / Total: {total:,} parameters")
    return model


# --- Dataset ---

def build_training_dataset(new_images, new_masks, new_labels,
                           replay_images, replay_labels,
                           augment_repeat, batch_size):
    """
    Build a single combined training dataset.
    New-class images are repeated `augment_repeat` times with bg-replacement augmentation.
    Replay images are included once with simple augmentation.
    Everything is shuffled together into one dataset.
    """
    n_new = len(new_images)
    n_replay = len(replay_images)
    n_new_aug = n_new * augment_repeat

    print(f"  New-class samples per epoch: {n_new_aug} ({n_new} x {augment_repeat})")
    print(f"  Replay samples per epoch:    {n_replay}")
    print(f"  Total samples per epoch:     {n_new_aug + n_replay}")

    # New-class dataset: repeat and augment with bg replacement
    new_ds = tf.data.Dataset.from_tensor_slices((new_images, new_masks, new_labels))
    new_ds = new_ds.repeat(augment_repeat)
    new_ds = new_ds.map(
        lambda img, msk, lbl: (augment_with_bg_replacement(img, msk), lbl),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Replay dataset: one pass, NO augmentation (anchors for existing classes)
    replay_ds = tf.data.Dataset.from_tensor_slices((replay_images, replay_labels))

    # Concatenate and shuffle everything together
    combined = new_ds.concatenate(replay_ds)
    combined = combined.shuffle(n_new_aug + n_replay, reshuffle_each_iteration=True)
    combined = combined.batch(batch_size)
    combined = combined.prefetch(tf.data.AUTOTUNE)
    return combined, n_new_aug + n_replay


# --- Training ---

def train(args):
    train_dir = os.path.join(args.train_data_dir, 'train')
    val_dir = os.path.join(args.train_data_dir, 'imagenet-val')

    # --- Load new-class training data ---
    print(f"\nScanning training data from {train_dir}...")
    new_class_train = scan_image_dir(train_dir, synset_filter=set(NEW_CLASS_SYNSETS))
    if not new_class_train:
        raise RuntimeError(f"No new-class training images found in {train_dir}")

    train_counts = Counter(cls for _, cls in new_class_train)
    for synset in NEW_CLASS_SYNSETS:
        idx = synset_to_index(synset)
        label = IMAGENET2012_CLASSES[synset]
        print(f"  {synset} ({label}): {train_counts.get(idx, 0)} training images")

    print(f"\nGenerating foreground masks (rembg)...")
    new_images, new_masks, new_labels = load_samples_with_masks(new_class_train)
    print(f"  {len(new_images)} images with masks")

    # --- Load replay data ---
    print(f"\nLoading replay data from {val_dir}...")
    all_existing = scan_image_dir(val_dir, synset_filter=set(SYNSET_IDS[:NUM_ORIGINAL_CLASSES]))
    random.seed(42)
    random.shuffle(all_existing)

    replay_counts = Counter()
    replay_samples = []
    for path, cls in all_existing:
        if replay_counts[cls] < args.replay_per_class:
            replay_samples.append((path, cls))
            replay_counts[cls] += 1

    print(f"  Sampling up to {args.replay_per_class} per class...")
    replay_images, replay_labels = load_samples(replay_samples)
    print(f"  {len(replay_images)} replay images across {len(replay_counts)} classes")

    # --- Load validation data (new classes only, NO augmentation) ---
    print(f"\nLoading validation data from {val_dir}...")
    new_class_val = scan_image_dir(val_dir, synset_filter=set(NEW_CLASS_SYNSETS))
    for synset in NEW_CLASS_SYNSETS:
        idx = synset_to_index(synset)
        count = sum(1 for _, c in new_class_val if c == idx)
        print(f"  {synset} ({IMAGENET2012_CLASSES[synset]}): {count} val images")

    val_images, val_labels = (load_samples(new_class_val) if new_class_val
                              else (None, None))

    # --- Build model ---
    print("\nBuilding model...")
    model = build_extended_model()

    # --- Build dataset ---
    print(f"\nBuilding training dataset (augment_repeat={args.augment_repeat})...")
    train_ds, total_samples = build_training_dataset(
        new_images, new_masks, new_labels,
        replay_images, replay_labels,
        args.augment_repeat, args.batch_size
    )

    # --- Compile ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- Train ---
    print(f"\nTraining: {args.epochs} epochs, lr={args.learning_rate}, "
          f"batch={args.batch_size}\n")
    model.fit(
        train_ds,
        epochs=args.epochs,
        verbose=1
    )

    # --- Validate ---
    print("\n" + "=" * 60)
    print("POST-TRAINING VALIDATION")
    print("=" * 60)

    if val_images is not None:
        _validate_new_classes(model, val_images, val_labels, val_dir)

    _spot_check_existing(model, val_dir, n_classes=15)

    # --- Save ---
    out_path = os.path.join(args.model_dir, args.output_model)
    print(f"\nSaving model to {out_path}...")
    model.save(out_path)
    print(f"Done.")


def _validate_new_classes(model, val_images, val_labels, val_dir):
    print("\nNew-class validation:")
    preds = model.predict(val_images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)

    val_files = []
    for synset in NEW_CLASS_SYNSETS:
        synset_dir = os.path.join(val_dir, synset)
        if not os.path.isdir(synset_dir):
            continue
        for fname in sorted(os.listdir(synset_dir)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                val_files.append((fname, synset))

    for synset in NEW_CLASS_SYNSETS:
        idx = synset_to_index(synset)
        mask = val_labels == idx
        if mask.sum() == 0:
            continue
        correct = (pred_classes[mask] == idx).sum()
        total = mask.sum()
        label = IMAGENET2012_CLASSES[synset]
        print(f"\n  {label} ({synset}): {correct}/{total} ({100 * correct / total:.1f}%)")

        for i in np.where(mask)[0]:
            top2 = np.argsort(preds[i])[-2:][::-1]
            top2_str = [f"{IMAGENET2012_CLASSES[SYNSET_IDS[t]]} ({100 * preds[i][t]:.1f}%)"
                        for t in top2]
            status = "✓" if pred_classes[i] == idx else "✗"
            fname = val_files[i][0] if i < len(val_files) else f"image_{i}"
            print(f"    {status} {fname}: {', '.join(top2_str)}")

    overall = (pred_classes == val_labels).mean()
    print(f"\n  Overall new-class accuracy: {100 * overall:.1f}%")


def _spot_check_existing(model, val_dir, n_classes=15):
    print(f"\nExisting-class spot check ({n_classes} random + coffee mug):")
    random.seed(123)
    check_synsets = random.sample(SYNSET_IDS[:NUM_ORIGINAL_CLASSES], min(n_classes, NUM_ORIGINAL_CLASSES))
    coffee = 'n03063599'
    if coffee in SYNSET_IDS and coffee not in check_synsets:
        check_synsets.append(coffee)

    for synset in check_synsets:
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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 with new classes")

    parser.add_argument('--train-data-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING') or '../data',
                        help="Root dir with train/ and imagenet-val/. Default: ../data")
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR') or '../models',
                        help="Output model directory. Default: ../models")
    parser.add_argument('--output-model', type=str, default='mobilenetv2-finetuned.keras',
                        help="Output filename. Default: mobilenetv2-finetuned.keras")

    parser.add_argument('--epochs', type=int, default=10,
                        help="Training epochs. Default: 10")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size. Default: 32")
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help="Learning rate. Default: 1e-4")
    parser.add_argument('--augment-repeat', type=int, default=20,
                        help="Augmentation repeat for new-class images. Default: 20")
    parser.add_argument('--replay-per-class', type=int, default=5,
                        help="Replay images per existing class. Default: 5")

    args, _ = parser.parse_known_args()

    print(f"Data root:    {args.train_data_dir}")
    print(f"Model dir:    {args.model_dir}")
    print(f"Output model: {args.output_model}")
    print(f"New classes:  {NEW_CLASS_SYNSETS}")

    train(args)


if __name__ == '__main__':
    main()

