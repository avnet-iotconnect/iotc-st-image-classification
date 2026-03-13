import os

# Install this package to suppress TF spam
try:
    import silence_tensorflow.auto
except ImportError: pass

import argparse

# You may want to uncomment this when using TFHub
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import os
import tensorflow as tf
import keras

MODEL_INPUT_SIZE = (224, 224)

def iotc_ota_send(args, file_path):
    """
    This function will send an OTA to the device with specified duid
    If a firmware for the device is not created, it will create one with a name based on the template name.
    """
    import avnet.iotconnect.restapi.lib.template as template
    from avnet.iotconnect.restapi.lib import firmware, upgrade, device, config, ota
    from avnet.iotconnect.restapi.lib import apiurl
    from avnet.iotconnect.restapi.lib import credentials


    if (args.iotc_env is not None or args.iotc_platform is not None or args.iotc_skey is not None
        or args.iotc_username is not None or args.iotc_password is not None
        ):
        if (args.iotc_env is None or args.iotc_platform is None or args.iotc_skey is None
                or args.iotc_username is None or args.iotc_password is None
            ):
            raise ValueError("All of the IoTConnect REST API authentication parameters must to be supplied.")
        config.env = args.iotc_env
        config.pf = args.iotc_platform
        config.skey = args.iotc_skey
        apiurl.configure_using_discovery()
        credentials.authenticate(username=args.iotc_username, password=args.iotc_password)
        print("Logged in successfully.")
    else:
        credentials.refresh()
        print("Credentials refreshed successfully successfully.")

    duid=args.send_to
    if duid is None:
        raise ValueError('iotc_send: The "duid" argument is required')
    print(f"Sending {file_path} to {duid}")
    device_object = device.get_by_duid(duid)
    template_object = template.get_by_guid(device_object.deviceTemplateGuid)
    firmware_guid = template_object.firmwareGuid

    if firmware_guid is None:
        import re
        firmware_name = template_object.templateName.upper() # first all letters to upper case
        firmware_name = re.sub(r'[^a-zA-Z0-9\s]', '', firmware_name) # remove any non-alpha-numeric characters
        firmware_name = firmware_name[:10] # use only up to 10 chars
        firmware_create_result = firmware.create(template_guid=template_object.guid, name=firmware_name, hw_version="1.0", initial_sw_version=None, description="Initial version", upgrade_description="New Model")
        firmware_upgrade_guid = firmware_create_result.firmwareUpgradeGuid
    else:
        firmware_upgrade_guid = upgrade.create(firmware_guid).newId
    upgrade.upload(firmware_upgrade_guid, file_path)
    upgrade.publish(firmware_upgrade_guid)
    ota.push_to_device(firmware_upgrade_guid, [device_object.guid])

def convert_to_tflite(model, calibration_data_path, per_tensor=True):
    def representative_dataset_synthetic():
        print("--- WARNING: USING SYNTHETIC REPRESENTATIVE DATASET ---")
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3) * 2.0 - 1.0
            yield [data.astype(np.float32)]

    def representative_dataset_gen():
        with np.load(calibration_data_path) as data:
            images = data[list(data.keys())[0]] # out key is usually "calibration_images"
            for i in range(len(images)):
                yield [images[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter.representative_dataset = representative_dataset_gen
    converter._experimental_disable_per_channel = per_tensor # if per_tensor, then disable
    return converter.convert()

def quantize(args):
    if args.input_model is None:
        default_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet'
        )
        default_model.build((None, 224, 224, 3))
        input_model = default_model
        input_model.save(os.path.join(args.model_dir, "base_model.keras"))
    else:
        print(f"Using {args.input_model}")
        input_model = keras.models.load_model(os.path.join(args.model_dir, args.input_model))

    calibration_data_file = os.path.join(args.train_data_dir, 'calibration.npz')

    out_file_path = os.path.join(args.model_dir, args.output_model)
    # unless forced, apply optimization only for per-channel quantization by default (unless explicitly disabled)
    optimize = args.force_optimization or (not args.no_optimization and not args.per_channel)
    if optimize:
        print("Applying the ST optimization to the model...")
        from st_optimization.model_formatting_ptq_per_tensor import model_formatting_ptq_per_tensor
        input_model = model_formatting_ptq_per_tensor(model_origin=input_model)

    print(f"Converting the model to {"per-channel" if args.per_channel else "per-tensor"} \"{out_file_path}\" with optimization={optimize}...")
    tflite_model = convert_to_tflite(input_model, calibration_data_file, per_tensor=not args.per_channel)

    # we have to do this warning after conversion as it will get lost in the log spam
    if not optimize and not args.per_channel:
        print("-----------------------------------")
        print("Warning: You are creating a per-tensor quantized model without applying the ST optimization. "
              "Most models like MobileNetV2 will produce a broken converted model as they are not suitable for per-tensor quantization!"
              )
        print("-----------------------------------")

    print("Writing the TFLite model...")
    with open(out_file_path, 'wb') as f:
        f.write(tflite_model)
    if args.send_to is not None:
        iotc_ota_send(args, out_file_path)
    print(f"Converted the model to {"per-channel" if args.per_channel else "per-tensor"} \"{out_file_path}\" with optimization={optimize}")

def main():
    parser = argparse.ArgumentParser()

    # SageMaker-provided arguments
    parser.add_argument('--train-data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING') if os.environ.get('SM_CHANNEL_TRAINING') is not None else '../data',
                        help="Location of calibration and other data. Default: ../data'")
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR') if os.environ.get('SM_MODEL_DIR') is not None else '../models',
                        help="Location where the models will be written or read. Default: '../models'")

    parser.add_argument('--input-model', type=str, default=None,
                        help="Optional model file from model-dir to load into the application. By default a model will be instantiated with Keras")
    parser.add_argument('--output-model', type=str, default="mobilenetv2-optimized.tflite",
                        help="File name to be written for the output model")
    parser.add_argument('--per-channel', action="store_true", default=False,
                        help="Specify this flag to quantize a per-channel model. Otherwise per-tensor model will be created.")
    parser.add_argument('--no-optimization', action="store_true", default=False,
                        help="Specify this flag to disable the per-tensor model optimization step.")
    parser.add_argument('--force-optimization', action="store_true", default=False,
                        help="Specify this flag to force the optimization on a per-channel model.")

    # IoTConnect OTA and user config:
    parser.add_argument('--send-to', type=str, default=None,
                        help="Optional name of your IoTConnect device to send the new model to")
    parser.add_argument('--iotc-username', type=str, default=os.environ.get('IOTC_USER'),
                        help="Your account username (email). IOTC_USER environment variable can be used instead.")
    parser.add_argument('--iotc-password', type=str, default=os.environ.get('IOTC_PASS'),
                        help="Your account password. IOTC_PASS environment variable can be used instead.")
    parser.add_argument('--iotc-platform', type=str, default=os.environ.get('IOTC_PF'),
                        help='Account platform ("aws" for AWS, or "az" for Azure). IOTC_PF environment variable can be used instead.')
    parser.add_argument('--iotc-env', type=str, default=os.environ.get('IOTC_ENV'),
                        help='Account environment - From settings -> Key Vault in the Web UI. IOTC_ENV environment variable can be used instead.'),
    parser.add_argument('--iotc-skey', type=str, default=os.environ.get('IOTC_SKEY'),
                        help="Your solution key. IOTC_SKEY environment variable can be used instead."),

    args, _ = parser.parse_known_args()

    print(f"Data will be read from: {args.train_data_dir}")
    print(f"Model will be saved to: {args.model_dir}")

    quantize(args)

    print(f"Saved models are at: {args.model_dir}")


if __name__ == '__main__':
    main()