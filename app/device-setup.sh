#!/bin/bash

set -e

apt update -y
apt install -y --upgrade python3 python3-pip python3-venv python3-wheel git cmake
apt install -y apt-openstlinux-x-linux-ai-npu x-linux-ai-tool
x-linux-ai --install python3-libstai-mpu
x-linux-ai --install libstai-mpu-ovx6
x-linux-ai --install python3-libtensorflow-lite
x-linux-ai --install libstai-mpu-tflite6
# application-resources should be all we need, but just in case install the full x-linux-ai-application
#x-linux-ai --install application-resources
x-linux-ai --install packagegroup-x-linux-ai-demo-npu

# /usr/local/x-linux-ai/resources/config_board_npu.sh
# I think it is needed for some demos
x-linux-ai --install config-npu

# packagegroup-x-linux-ai-npu or...
# packagegroup-x-linux-ai-npu
# or packagegroup-x-linux-ai-tflite-npu
packagegroup-x-linux-ai-demo-npu


wget -q -O ImageNetLabels.txt https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt

wget -q -O mobilenet_v2_1.0_224_int8_per_tensor.nb https://github.com/STMicroelectronics/meta-st-x-linux-ai/raw/refs/heads/main/recipes-samples/image-classification/models/files/mobilenet_v2_1.0_224_int8_per_tensor.nb
wget -q -O mobilenet_v2_1.0_224_int8_per_tensor.tflite https://github.com/STMicroelectronics/meta-st-x-linux-ai/raw/refs/heads/main/recipes-samples/image-classification/models/files/mobilenet_v2_1.0_224_int8_per_tensor.tflite

python3 -m venv --system-site-packages ~/.venv-staiicdemo
source ~/.venv-staiicdemo/bin/activate
python3 -m pip install -r requirements.txt


unzip -o *-certificates.zip
mv cert_*.crt device-cert.pem
mv pk_*.pem device-pkey.pem

echo "Execute the app with :"
echo "source ~/.venv-staiicdemo/bin/activate"
echo "python3 app.py -m mobilenet_v2_1.0_224_int8_per_tensor.nb"
