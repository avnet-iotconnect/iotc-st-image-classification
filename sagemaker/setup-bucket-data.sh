#!/bin/bash

set -e
set -x

cd "$(dirname "$0")"

bucket=$(./get-default-bucket.py)
aws s3 mb "s3://${bucket}" --region "${AWS_REGION}" >dev/null 2>/dev/null || true


AWS_REGION=$(aws configure get region)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SAGEMAKER_ROLE_NAME="sagemaker-execution-role-${ACCOUNT_ID}" # Consistent naming
S3_POLICY_NAME="SageMakerDataPolicy-${bucket}"

read -r -d '@@@' TRUST_POLICY_JSON <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
@@@
EOF

# Define the S3 Inline Policy Template (will be modified with bucket name)
read -r -d '@@@' S3_POLICY_JSON <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${bucket}",
                "arn:aws:s3:::${bucket}/*"
            ]
        }
    ]
}
@@@
EOF

if ! aws iam get-role --role-name "${SAGEMAKER_ROLE_NAME}" &>/dev/null; then
    aws iam create-role \
      --role-name "${SAGEMAKER_ROLE_NAME}" \
      --assume-role-policy-document "${TRUST_POLICY_JSON}"

    aws iam attach-role-policy \
      --role-name "${SAGEMAKER_ROLE_NAME}" \
      --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"

    echo "Waiting for IAM role propagation..."
    sleep 10 # wait a bit for role to appear

fi

aws iam put-role-policy \
  --role-name "${SAGEMAKER_ROLE_NAME}" \
  --policy-name "${S3_POLICY_NAME}" \
  --policy-document "${S3_POLICY_JSON}" \
  >/dev/null

aws s3 cp ../data/calibration.npz "s3://${bucket}"/quantization/data/calibration.npz
aws s3 cp ../data/calibration-stedgeai.npz "s3://${bucket}"/quantization/data/calibration-stedgeai.npz




