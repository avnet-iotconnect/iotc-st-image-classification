#!/bin/bash

set -e

apt update -y
apt install -y --upgrade python3 python3-pip python3-venv python3-wheel git cmake
apt install -y apt-openstlinux-x-linux-ai-npu x-linux-ai-tool
x-linux-ai --install python3-libstai-mpu
x-linux-ai --install libstai-mpu-ovx6
x-linux-ai --install python3-libtensorflow-lite
x-linux-ai --install libstai-mpu-tflite6
x-linux-ai --install packagegroup-x-linux-ai-demo-npu
x-linux-ai --install config-npu

# The user should download labels from the project app directory with new IDs
# wget -q -O ImageNetLabels.txt https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt

wget -q -O mobilenet_v2_1.0_224_int8_per_tensor.nb https://github.com/STMicroelectronics/meta-st-x-linux-ai/raw/refs/heads/main/recipes-samples/image-classification/models/files/mobilenet_v2_1.0_224_int8_per_tensor.nb
wget -q -O mobilenet_v2_1.0_224_int8_per_tensor.tflite https://github.com/STMicroelectronics/meta-st-x-linux-ai/raw/refs/heads/main/recipes-samples/image-classification/models/files/mobilenet_v2_1.0_224_int8_per_tensor.tflite
wget -q -O mobilenetv2-finetuned-pc.nb https://downloads.iotconnect.io/ai-data/mobilenet-v2/mobilenetv2-finetuned-pc.nb
wget -q -O mobilenetv2-finetuned-pc.tflite https://downloads.iotconnect.io/ai-data/mobilenet-v2/mobilenetv2-finetuned-pc.tflite

echo Creating Python virtual environment and installing dependencies...
python3 -m venv --system-site-packages ~/.venv-staiicdemo
source ~/.venv-staiicdemo/bin/activate
python3 -m pip install -r requirements.txt


unzip -o *-certificates.zip
mv cert_*.crt device-cert.pem
mv pk_*.pem device-pkey.pem

echo "----------------------------------------------"
echo "Execute the app with :"
echo "source ~/.venv-staiicdemo/bin/activate"
echo "python3 app.py -m mobilenet_v2_1.0_224_int8_per_tensor.nb"
