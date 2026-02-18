#!/bin/bash

set -e

pushd ~/app/

source ~/python-samples-for-amazon-kinesis-video-streams-with-webrtc/.venv/bin/activate

source ~/.aws-env.sh

python3 kvsWebRTCClientMaster.py --channel-arn 'arn:aws:kinesisvideo:us-east-1:857898724229:channel/andraka-ameba-channel/1768592534309'

popd