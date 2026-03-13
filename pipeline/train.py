"""
Simple fine-tuning of MobileNetV2: add new classes (1000 -> 1003).
No rembg, no background replacement. Just standard image augmentation.
"""

import os
import sys

try:
    import silence_tensorflow.auto
except ImportError:
    pass

import random
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classes import IMAGENET2012_CLASSES

# ── Config ──────────────────────────────────────────────────
DATA_ROOT       = '../data'
MODEL_DIR       = '../models'
OUTPUT_MODEL    = 'mobilenetv2-finetuned.keras'

EPOCHS          = 10
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
AUGMENT_REPEAT  = 20      # how many augmented copies per new-class image
REPLAY_PER_CLASS = 5      # images per existing class for replay

MODEL_INPUT_SIZE = (224, 224)
NUM_ORIGINAL     = 1000
IMAGE_EXTS       = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}

SYNSET_IDS       = list(IMAGENET2012_CLASSES.keys())
NUM_TOTAL        = len(SYNSET_IDS)
NEW_SYNSETS      = [s for s in SYNSET_IDS if s.startswith("n9999")]


# ── Helpers ─────────────────────────────────────────────────

def synset_to_index(synset_id):
    return SYNSET_IDS.index(synset_id)


def load_image(path):
    img = Image.open(path).convert('RGB').resize(MODEL_INPUT_SIZE)
    return keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype=np.float32)
    )


def scan_dir(base, synset_filter=None):
    out = []
    if not os.path.isdir(base):
        return out
    for sid in sorted(os.listdir(base)):
        d = os.path.join(base, sid)
        if not os.path.isdir(d) or sid not in SYNSET_IDS:
            continue
        if synset_filter and sid not in synset_filter:
            continue
        idx = synset_to_index(sid)
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append((os.path.join(d, f), idx))
    return out


def load_all(samples):
    imgs, lbls = [], []
    for p, c in samples:
        try:
            imgs.append(load_image(p))
            lbls.append(c)
        except Exception as e:
            print(f"  skip {p}: {e}")
    return np.stack(imgs), np.array(lbls, dtype=np.int32)


# ── Augmentation (plain, no masks) ──────────────────────────

def augment(img):
    """Standard augmentation for a single [224,224,3] image in [-1,1]."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    # random crop 80-100% then resize back
    frac = tf.random.uniform([], 0.8, 1.0)
    sz = tf.cast(224.0 * frac, tf.int32)
    img = tf.image.random_crop(img, [sz, sz, 3])
    img = tf.image.resize(img, MODEL_INPUT_SIZE)
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img


# ── Model ───────────────────────────────────────────────────

def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=True, weights='imagenet'
    )
    orig_dense = base.get_layer('predictions')
    orig_w, orig_b = orig_dense.get_weights()

    # Chop off original head, keep up to GlobalAveragePooling2D
    features = keras.Model(base.input, base.layers[-2].output)
    for l in features.layers:
        l.trainable = False

    x = features.output
    out = keras.layers.Dense(NUM_TOTAL, activation='softmax', name='predictions')(x)
    model = keras.Model(features.input, out)

    # Copy original weights into first 1000 columns, random-init new ones
    dense = model.get_layer('predictions')
    w = np.zeros((1280, NUM_TOTAL), dtype=np.float32)
    b = np.zeros(NUM_TOTAL, dtype=np.float32)
    w[:, :NUM_ORIGINAL] = orig_w
    b[:NUM_ORIGINAL]    = orig_b
    rng = np.random.default_rng(42)
    w[:, NUM_ORIGINAL:]  = rng.normal(0, 0.01, (1280, NUM_TOTAL - NUM_ORIGINAL))
    b[NUM_ORIGINAL:]     = 0.0
    dense.set_weights([w, b])

    print(f"Model: {NUM_ORIGINAL} -> {NUM_TOTAL} classes")
    return model


# ── Dataset ─────────────────────────────────────────────────

def build_dataset(new_imgs, new_lbls, replay_imgs, replay_lbls):
    n_new = len(new_imgs) * AUGMENT_REPEAT
    n_rep = len(replay_imgs)
    print(f"  New-class: {n_new} ({len(new_imgs)} x {AUGMENT_REPEAT})")
    print(f"  Replay:    {n_rep}")
    print(f"  Total:     {n_new + n_rep}")

    new_ds = tf.data.Dataset.from_tensor_slices((new_imgs, new_lbls))
    new_ds = new_ds.repeat(AUGMENT_REPEAT)
    new_ds = new_ds.map(lambda x, y: (augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    rep_ds = tf.data.Dataset.from_tensor_slices((replay_imgs, replay_lbls))
    # No augmentation on replay — they are anchors

    ds = new_ds.concatenate(rep_ds)
    ds = ds.shuffle(n_new + n_rep, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Validation ──────────────────────────────────────────────

def validate_new(model, val_imgs, val_lbls, val_dir):
    print("\nNew-class validation:")
    preds = model.predict(val_imgs, verbose=0)
    pred_cls = np.argmax(preds, axis=1)

    # build file list for per-image reporting
    files = []
    for s in NEW_SYNSETS:
        d = os.path.join(val_dir, s)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                files.append(f)

    for s in NEW_SYNSETS:
        idx = synset_to_index(s)
        mask = val_lbls == idx
        if not mask.any():
            continue
        ok = (pred_cls[mask] == idx).sum()
        tot = mask.sum()
        print(f"\n  {IMAGENET2012_CLASSES[s]} ({s}): {ok}/{tot} ({100*ok/tot:.1f}%)")
        for i in np.where(mask)[0]:
            top2 = np.argsort(preds[i])[-2:][::-1]
            t2 = ", ".join(f"{IMAGENET2012_CLASSES[SYNSET_IDS[t]]} ({100*preds[i][t]:.1f}%)" for t in top2)
            mark = "✓" if pred_cls[i] == idx else "✗"
            fn = files[i] if i < len(files) else f"img_{i}"
            print(f"    {mark} {fn}: {t2}")
    print(f"\n  Overall: {100*(pred_cls == val_lbls).mean():.1f}%")


def spot_check(model, val_dir, n=15):
    print(f"\nExisting-class spot check ({n} random + coffee mug):")
    random.seed(123)
    existing = [s for s in SYNSET_IDS if not s.startswith("n9999")]
    picks = random.sample(existing, min(n, len(existing)))
    coffee = 'n03063599'
    if coffee in SYNSET_IDS and coffee not in picks:
        picks.append(coffee)
    for s in picks:
        samples = scan_dir(val_dir, {s})[:10]
        if not samples:
            continue
        imgs, lbls = load_all(samples)
        p = np.argmax(model.predict(imgs, verbose=0), axis=1)
        ok = (p == lbls).sum()
        print(f"  {IMAGENET2012_CLASSES[s]}: {ok}/{len(lbls)} ({100*ok/len(lbls):.1f}%)")


# ── Main ────────────────────────────────────────────────────

def main():
    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir   = os.path.join(DATA_ROOT, 'imagenet-val')

    # --- new-class training images (no masks, no rembg) ---
    print(f"\nScanning {train_dir}...")
    new_samples = scan_dir(train_dir, set(NEW_SYNSETS))
    counts = Counter(c for _, c in new_samples)
    for s in NEW_SYNSETS:
        print(f"  {s} ({IMAGENET2012_CLASSES[s]}): {counts.get(synset_to_index(s), 0)} imgs")
    new_imgs, new_lbls = load_all(new_samples)
    print(f"  Loaded {len(new_imgs)} new-class images")

    # --- replay ---
    print(f"\nLoading replay from {val_dir}...")
    existing = [s for s in SYNSET_IDS if not s.startswith("n9999")]
    all_exist = scan_dir(val_dir, set(existing))
    random.seed(42)
    random.shuffle(all_exist)
    rc = Counter()
    rsamples = []
    for p, c in all_exist:
        if rc[c] < REPLAY_PER_CLASS:
            rsamples.append((p, c))
            rc[c] += 1
    replay_imgs, replay_lbls = load_all(rsamples)
    print(f"  {len(replay_imgs)} replay images across {len(rc)} classes")

    # --- validation ---
    print(f"\nLoading validation...")
    val_samples = scan_dir(val_dir, set(NEW_SYNSETS))
    for s in NEW_SYNSETS:
        cnt = sum(1 for _, c in val_samples if c == synset_to_index(s))
        print(f"  {s} ({IMAGENET2012_CLASSES[s]}): {cnt} val imgs")
    val_imgs, val_lbls = load_all(val_samples) if val_samples else (None, None)

    # --- model ---
    print("\nBuilding model...")
    model = build_model()

    # --- dataset ---
    print(f"\nBuilding dataset...")
    ds = build_dataset(new_imgs, new_lbls, replay_imgs, replay_lbls)

    # --- train ---
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\nTraining: {EPOCHS} epochs, lr={LEARNING_RATE}, batch={BATCH_SIZE}\n")
    model.fit(ds, epochs=EPOCHS, verbose=1)

    # --- validate ---
    print("\n" + "=" * 60)
    print("POST-TRAINING VALIDATION")
    print("=" * 60)
    if val_imgs is not None:
        validate_new(model, val_imgs, val_lbls, val_dir)
    spot_check(model, val_dir)

    # --- save ---
    out = os.path.join(MODEL_DIR, OUTPUT_MODEL)
    print(f"\nSaving to {out}...")
    model.save(out)
    print("Done.")


if __name__ == '__main__':
    main()

