import os
import urllib.parse
import tarfile
import sagemaker
import boto3

from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()


bucket = sagemaker_session.default_bucket() # Or your specific bucket
s3_data_path = f's3://{bucket}/data/calibration.npz'
s3_output_path = f's3://{bucket}/output/'


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

def get_sagemaker_execution_role_arn_by_pattern():
    """
    Discovers the SageMaker execution role ARN based on the expected naming pattern.
    Assumes the role name is 'sagemaker-execution-role-{AWS_ACCOUNT_ID}'.
    Fails if the role or account ID cannot be determined.
    """
    sts_client = boto3.client('sts')
    account_id = sts_client.get_caller_identity()['Account']

    iam_client = boto3.client('iam')
    expected_role_name = f"sagemaker-execution-role-{account_id}"

    response = iam_client.get_role(RoleName=expected_role_name)
    return response['Role']['Arn']

def copy_to_local(args, estimator):
    parsed_url = urllib.parse.urlparse(estimator.model_data)

    s3 = boto3.client('s3')
    try:
        model_tar = '/tmp/model.tar.gz'
        s3.download_file(parsed_url.netloc, parsed_url.path.lstrip('/'), model_tar)
        with tarfile.open(model_tar, 'r:gz') as tar:
            # For Python < 3.14
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=args.model_dir, filter='data', verbose=True)
            else:
                tar.extractall(path=args.model_dir)
    except Exception as e:
        print(f"Error downloading file: {e}")

def parse_hyperparameters_and_args():
    import argparse
    parser = argparse.ArgumentParser()

    # SageMaker-provided arguments

    # Compared to the invoked script, do not use train and model dir. We want this to be taken in at runtime
    # parser.add_argument('--train-data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING') if os.environ.get('SM_CHANNEL_TRAINING') is not None else '../data')

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR') if os.environ.get('SM_MODEL_DIR') is not None else '../models')

    parser.add_argument('--base-model', type=str, default=None)
    parser.add_argument('--input-model', type=str, default=None)
    parser.add_argument('--output-model', type=str, default="mobilenetv2-sm-optimized.tflite")
    parser.add_argument('--per-channel', action="store_true", default=False)

    # IoTConnect OTA and user config:
    parser.add_argument('--send-to', type=str, default=None)
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

    hyperparameters = {}
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:  # Skip None values (optional params)
            hyperparameters[arg_name.replace("_", "-")] = str(arg_value)

    # This parameter will not be passed. Let sagemaker do it internally
    # Used only for local path
    if hyperparameters.get('model-dir') is not None:
        del hyperparameters['model-dir']

    # special handling for optional parameters that have action="store_true"
    if args.per_channel:
        hyperparameters['per-channel'] = ""
    else:
        del hyperparameters['per-channel']

    # not supported directly from sagemaker because it does not support python 3.11 (minimum for our rest API)
    if hyperparameters.get('send-to') is not None:
        del hyperparameters['send-to']

    # remove all IoTConnect related parameters as well
    hyperparameters = {key: value for key, value in hyperparameters.items() if not key.startswith('iotc-')}

    return hyperparameters, args

def main():

    role = get_sagemaker_execution_role_arn_by_pattern()

    hyperparameters, args = parse_hyperparameters_and_args()
    estimator = TensorFlow(
        entry_point='../pipeline/quantize.py',
        # No source_dir — avoids pipeline/requirements.txt being auto-installed
        # by the training toolkit (which would upgrade TF and break the container).
        # The container already ships tensorflow, numpy, and pillow.
        dependencies=['../pipeline/st_optimization'],
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge', # or 'ml.g4dn.xlarge'
        framework_version='2.18', # TensorFlow version
        py_version='py310',
        hyperparameters=hyperparameters,
        output_path=s3_output_path, # Where the trained model will be saved
        model_dir='/opt/ml/model'  # We want to save the model locally and some other stuff
    )

    inputs = {
        'training': sagemaker.TrainingInput(s3_data_path, distribution='FullyReplicated',
                                            content_type='application/tar+gzip')
    }

    estimator.fit(inputs)
    print(f"Trained model saved to: {estimator.model_data}")

    copy_to_local(args, estimator)
    print(f"Model files have been copied locally.")

    out_file_path = os.path.join(args.model_dir, args.output_model)

    if args.send_to is not None:
        iotc_ota_send(args, out_file_path)

    print("Done.")

if __name__ == '__main__':
    main()