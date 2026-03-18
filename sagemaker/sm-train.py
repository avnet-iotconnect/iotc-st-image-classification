import os
import urllib.parse
import tarfile
import sagemaker
import boto3

from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()  # Or your specific bucket
s3_data_path = f's3://{bucket}/data/'
s3_output_path = f's3://{bucket}/output/'


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
            tar.list(tarfile)
            # For Python < 3.14
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=args.model_dir, filter='data')
            else:
                tar.extractall(path=args.model_dir)
    except Exception as e:
        print(f"Error downloading file: {e}")


def parse_hyperparameters_and_args():
    import argparse
    parser = argparse.ArgumentParser()

    # Compared to the invoked script, do not use train-data-dir and model-dir.
    # We want these to be taken in at runtime by SageMaker.
    # parser.add_argument('--train-data-dir', ...)

    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR') if os.environ.get('SM_MODEL_DIR') is not None else '../models',
                        help="Local directory where the output model will be downloaded to. Default: '../models'")

    parser.add_argument('--output-model', type=str, default='mobilenetv2-sm-finetuned.keras',
                        help="File name to be written for the output model")

    args, _ = parser.parse_known_args()

    hyperparameters = {}
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:  # Skip None values (optional params)
            hyperparameters[arg_name.replace("_", "-")] = str(arg_value)

    # This parameter will not be passed. Let sagemaker provide it internally.
    # Used only for local path.
    if hyperparameters.get('model-dir') is not None:
        del hyperparameters['model-dir']

    return hyperparameters, args


def main():

    role = get_sagemaker_execution_role_arn_by_pattern()

    hyperparameters, args = parse_hyperparameters_and_args()
    estimator = TensorFlow(
        entry_point='../pipeline/train.py',
        # No source_dir — avoids pipeline/requirements.txt being auto-installed
        # by the training toolkit (which would upgrade TF and break the container).
        # The container already ships tensorflow, numpy, and pillow.
        # classes.py is needed by train.py for IMAGENET2012_CLASSES.
        dependencies=['../pipeline/classes.py'],
        role=role,
        instance_count=1,
        instance_type='ml.g4dn.xlarge',  # T4 GPU — cheapest GPU instance, ~10-20x faster than m5 CPU for CNN training
        framework_version='2.18',  # TensorFlow version
        py_version='py310',
        hyperparameters=hyperparameters,
        output_path=s3_output_path,
        model_dir='/opt/ml/model',  # We want to save the model locally and some other stuff
        # The training data is ~7GB. Default volume is 30GB which should be sufficient,
        # but increase if needed:
        # volume_size=50,
    )

    # Use S3Prefix to download the entire recursive s3://bucket/data/ tree
    # into the container's /opt/ml/input/data/training/ directory.
    # SageMaker will preserve the directory structure, so train.py will see
    # train/ and imagenet-val/ subdirectories under SM_CHANNEL_TRAINING.
    inputs = {
        'training': sagemaker.TrainingInput(
            s3_data_path,
            distribution='FullyReplicated',
            s3_data_type='S3Prefix',
            input_mode='File',
        )
    }

    estimator.fit(inputs)
    print(f"Trained model saved to: {estimator.model_data}")

    copy_to_local(args, estimator)
    print(f"Model files have been copied locally.")

    out_file_path = os.path.join(args.model_dir, args.output_model)
    print(f"Output model: {out_file_path}")

    print("Done.")


if __name__ == '__main__':
    main()


