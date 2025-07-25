import os
# You may want to uncomment this when using TFHub
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse

import random

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_INPUT_SIZE = (224, 224)


def convert_to_tflite_mpx(model, calibration_data_path, per_tensor=True):
    def representative_dataset_gen():
        with np.load(calibration_data_path) as data:
            images = data[list(data.keys())[0]] # out key is usually "calibration_images"
            for i in range(len(images)):
                yield [images[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,  # keep this first
    #     tf.lite.OpsSet.SELECT_TF_OPS  # let TF kernels run the rest
    # ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    converter.representative_dataset = representative_dataset_gen

    converter._experimental_disable_per_channel = per_tensor # if per_tensor, then disable

    if per_tensor:
        x=1
        # Optional: Explicitly allow asymmetric activations (default behavior)
        # converter.experimental_new_dynamic_range_quantizer = False  # <== restores pre-2.18 numerics
        # converter._experimental_full_integer_quantization = True
        # converter.use_symmetric_quantization = False  # Not strictly needed (asymmetric is default)
        # converter.use_weights_symmetric_quantization = False  # Allow weight zero-point != 0

    return converter.convert()

def quantize(args):
    if args.input_model is None:

        def qat():
            import tensorflow_model_optimization as tfmot
            # 1. build a plain Keras model (logits only)
            inputs = tf.keras.layers.Input(shape=(224, 224, 3))
            logits = hub.KerasLayer(
                "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
                trainable=True)(inputs)
            model = tf.keras.Model(inputs, logits)

            # 2. one-line QAT wrapper
            q_aware_model = tfmot.quantization.keras.quantize_model(model)

            # 3. fine-tune for **a few epochs** on ImageNet or on your own labelled data
            q_aware_model.compile(optimizer='adam',
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

            npz=np.load("../data/calibration.npz")
            images = npz["representative_data"]
            labels = npz["labels"]
            ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(1)
            q_aware_model.fit(ds, epochs=3)  # 3–5 epochs is usually enough

            # 4. add Softmax and export per-tensor INT8
            final = tf.keras.Sequential([q_aware_model, tf.keras.layers.Softmax()])
            final.build((None, 224, 224, 3))
            return final

        # input_model = tf.keras.applications.MobileNetV2(
        #     input_shape=(224, 224, 3),
        #     alpha=1.0,
        #     include_top=True,
        #     weights='imagenet'
        # )
        import tensorflow_hub as hub
        url="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
        url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
        url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/3"
        url="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
        input_model = tf.keras.Sequential([
            hub.KerasLayer(url, trainable=False),
            tf.keras.layers.Softmax()  # Make it the same as the base model - max to 1.
        ])
        input_model.build((None, 224, 224, 3))
        #
        # from qat_helper import apply_qat_one_epoch
        #
        # input_model = apply_qat_one_epoch(
        #     base_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
        #     npz_path="../data/calibration.npz"
        # )

    else:
        print(f"Using {args.input_model}")
        input_model = tf.keras.models.load_model(os.path.join(args.model_dir, args.input_model))

    calibration_data_file = os.path.join(args.train_data_dir, 'calibration.npz')

    out_file_path = os.path.join(args.model_dir, args.output_model)
    print(f"Converting to {out_file_path} per_tensor={str(args.per_tensor)}")
    tflite_model = convert_to_tflite_mpx(input_model, calibration_data_file, per_tensor=args.per_tensor)
    print("Writing...")
    with open(out_file_path, 'wb') as f:
        f.write(tflite_model)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker-provided arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR') if os.environ.get('SM_MODEL_DIR') is not None else '../models')
    parser.add_argument('--train_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING') if os.environ.get('SM_CHANNEL_TRAINING') is not None else '../data')
    parser.add_argument('--input_model', type=str, default=None)
    parser.add_argument('--output_model', type=str, default="quantized-model.tflite")
    parser.add_argument('--per_tensor', action="store_true", default=False)
    args, _ = parser.parse_known_args()

    print(f"Data will be read from: {args.train_data_dir}")
    print(f"Model will be saved to: {args.model_dir}")

    quantize(args)

    print(f"Saved models are at: {args.model_dir}")
