import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import keras.models
from PIL import Image

import functions

def mobilenetv2_validation_images(images_base, names, skip=0, max_images=100):
    val_imgs = []
    for name in names:
        start_directory = images_base + '/' + functions.to_synset_id(functions.normalized_name(name))
        count = max_images
        skip_count = skip
        got_one = False
        for root, dirs, files in os.walk(start_directory):
            for file in files:
                if skip_count > 0:
                    skip_count -= 1
                else:
                    if count == 0: break
                    count -= 1
                    got_one = True
                    val_imgs.append((root + '/' + file, name))
        if not got_one:
            raise RuntimeError(f"Got empty set from {start_directory}")

    return val_imgs

def run_mp2(args):
    custom_test_data = mobilenetv2_validation_images(args.train_data_dir + '/custom-data/test', ['yorkshire_terrier'],  max_images=100)
    extra_test_data = mobilenetv2_validation_images(
        args.train_data_dir + '/imagenet-val',
        ['yorkshire_terrier', 'norfolk_terrier', 'silky_terrier', 'australian_terrier'],
        skip=30,
        max_images=10
    )
    try:
        with open(args.model_dir + '/mobilenet_v2_1.0_224_int8_per_tensor.tflite', 'rb') as f:
            tflite_model_st = f.read()
        print(f"\n=== ST's Model (inference_tflite) ===")
        functions.inference_tflite(tflite_model_st, extra_test_data + custom_test_data, skip_names='mux')
        print(f"\n=== ST's Model (stai_inference) ===")
        functions.stai_inference(args.model_dir + '/mobilenet_v2_1.0_224_int8_per_tensor.tflite', extra_test_data + custom_test_data, skip_names='mux')
    except Exception:
        pass

    functions.stai_inference(args.model_dir + '/custom-mux-pc-u8f32.tflite', extra_test_data + custom_test_data, skip_names = 'mux')
    functions.stai_inference(args.model_dir + '/custom-mux-pc-u8f32.nb', extra_test_data + custom_test_data, skip_names = 'mux')
    functions.stai_inference(args.model_dir + '/custom-mux-st-pc-u8f32.nb', extra_test_data + custom_test_data, skip_names = 'mux')
    functions.stai_inference(args.model_dir + '/custom-mux-pt.tflite', extra_test_data + custom_test_data, skip_names='mux')
    functions.stai_inference(args.model_dir + '/custom-mux-pt.nb', extra_test_data + custom_test_data, skip_names='mux')
    functions.stai_inference(args.model_dir + '/custom-mux-st-pt-u8f32.nb', extra_test_data + custom_test_data, skip_names = 'mux')

def process_tflite_model(args, input_model, tflite_file_path, calibration_data='default', type='cpu:int8-int8'):
    if calibration_data == 'default':
        calibration_data = args.train_data_dir + '/calibration.npz'

    if args.demo and os.path.isfile(tflite_file_path):
        with open(tflite_file_path, 'rb') as f:
            tflite_model = f.read()
    else:
        print("Converting to tflite...")
        tflite_model = functions.convert_to_tflite(input_model, calibration_data, type=type)
        print("Done.")
        with open(tflite_file_path, 'wb') as f:
            f.write(tflite_model)
    return tflite_model

def run(args):
    is_in_sagemaker = os.environ.get('SM_CHANNEL_TRAINING') is not None
    if is_in_sagemaker:
        import tarfile # builtin
        def extract_tarball(tarball_path, output_dir):
            with tarfile.open(tarball_path, 'r:gz') as tar:
                tar.extractall(path=output_dir)
        extract_tarball(args.train_data_dir + '/images.tgz', args.train_data_dir)

    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights='imagenet'
    )

    # 1. FREEZE EVERYTHING FIRST
    for layer in model.layers:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])


    retrained_model_h5 = args.model_dir + '/custom-mux.h5'
    if not args.demo or not os.path.isfile(retrained_model_h5):
        custom_train_data = mobilenetv2_validation_images(args.train_data_dir + '/custom-data/train', ['yorkshire_terrier'])
        custom_validation_data = mobilenetv2_validation_images(args.train_data_dir + '/custom-data/validation', ['yorkshire_terrier'])
        extra_train_data = mobilenetv2_validation_images(
            args.train_data_dir + '/imagenet-val',
            ['yorkshire_terrier', 'norfolk_terrier', 'silky_terrier', 'australian_terrier'],
            skip=0,
            max_images=10
        )
        extra_validation_data = mobilenetv2_validation_images(
            args.train_data_dir + '/imagenet-val',
            ['yorkshire_terrier', 'norfolk_terrier', 'silky_terrier', 'australian_terrier'],
            skip=10,
            max_images=20
        )
        retrained_model, history = functions.fit_mobilenetv2_old(
            model,
            custom_train_data + extra_train_data,  # Now NumPy arrays/tensors
            custom_validation_data + extra_validation_data,  # Now NumPy arrays/tensors
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        print(f"Saving to {retrained_model_h5}")
        retrained_model.save(retrained_model_h5)
        if is_in_sagemaker:
            # Otherwise there's nothing in output. This breaks if ran locally
            tf.saved_model.save(retrained_model, os.path.join(args.model_dir, "model"))
    else:
        retrained_model = keras.models.load_model(retrained_model_h5)

    # gets rid of some warnings
    retrained_model.compile(metrics=['accuracy'])

    tflite_model = process_tflite_model(args, retrained_model, args.model_dir + '/custom-mux.tflite')
    tflite_model_st_pt_u8f32 = process_tflite_model(args, retrained_model, args.model_dir + '/custom-mux-pt-u8f32.tflite', type='st-npu:uint8-float32')
    tflite_model_st_pc_u8f32 = process_tflite_model(args, retrained_model, args.model_dir + '/custom-mux-pc-u8f32.tflite', type='st-npu-pc:uint8-float32')

    # tflite_model_synth = process_tflite_model(args, retrained_model, args.model_dir + '/custom-mux-synth-calib.tflite')

    custom_test_data = mobilenetv2_validation_images(args.train_data_dir + '/custom-data/test', ['yorkshire_terrier'],  max_images=100)
    extra_test_data = mobilenetv2_validation_images(
        args.train_data_dir + '/imagenet-val',
        ['yorkshire_terrier', 'norfolk_terrier', 'silky_terrier', 'australian_terrier'],
        skip=30,
        max_images=10
    )


    print(f"\n=== Fine Tuned ===")
    functions.inference_h5_model(retrained_model, extra_test_data + custom_test_data, skip_names='mux')

    # print(f"\n=== TFLite (Synthetic Calibration) ===")
    # functions.inference_tflite(tflite_model_synth, extra_test_data + custom_test_data, skip_names='mux')

    with open(args.model_dir + '/mobilenet_v2_1.0_224_int8_per_tensor.tflite', 'rb') as f:
        tflite_model_st_pt_210 = f.read()
    print(f"\n=== ST's Quantized uint8/float32 Model per-tensor ===")
    functions.inference_tflite(tflite_model_st_pt_210, extra_test_data + custom_test_data, skip_names='mux')

    print(f"\n=== TFLite uint8/float32 (Per Tensor) ===")
    functions.inference_tflite(tflite_model_st_pt_u8f32, extra_test_data + custom_test_data, skip_names='mux')

    print(f"\n=== TFLite uint8/float32 (Per Channel) ===")
    functions.inference_tflite(tflite_model_st_pc_u8f32, extra_test_data + custom_test_data, skip_names='mux')

    print(f"\n=== TFLite int8/int8 (Per Channel) ===")
    functions.inference_tflite(tflite_model, extra_test_data + custom_test_data, skip_names='mux')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker-provided arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR') if os.environ.get('SM_MODEL_DIR') is not None else '../models')
    parser.add_argument('--train_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING') if os.environ.get('SM_CHANNEL_TRAINING') is not None else '../data')
    parser.add_argument('--tf_model_file_name', type=str, default=None)
    parser.add_argument('--per_channel', type=bool, default=True)
    args, _ = parser.parse_known_args()

    print(f"Data will be read from: {args.train_data_dir}")
    print(f"Model will be saved to: {args.model_dir}")

    if not args.run_mp2:
        run(args)
    else:
        run_mp2(args)

    print(f"Saved models are at: {args.model_dir}")
