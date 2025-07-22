import sagemaker
import boto3

from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()


bucket = sagemaker_session.default_bucket() # Or your specific bucket
s3_data_path = f's3://{bucket}/data/images.tgz'
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

role = get_sagemaker_execution_role_arn_by_pattern()

estimator = TensorFlow(
    entry_point='train.py',
    source_dir='../training', # Directory containing train.py and requirements.txt
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge', # 'ml.m5.xlarge'
    framework_version='2.18', # TensorFlow version
    py_version='py310',
    hyperparameters={
        'learning_rate': 0.0001,
        'epochs': 30
    },
    output_path=s3_output_path, # Where the trained model will be saved
    model_dir='/opt/ml/model'  # We want to save the model locally and some other stuff
)

inputs = {
    'training': sagemaker.TrainingInput(s3_data_path, distribution='FullyReplicated',
                                        content_type='application/tar+gzip')
}

estimator.fit(inputs)
print(f"Trained model saved to: {estimator.model_data}")