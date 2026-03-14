# Introduction
This is Avnet's IoTConnect Image Classification Demo for STM23 MP2, with AWS Sagemaker 
and OTA updates support.

This project demonstrates how a Tensorflow base model can be converted to TFLite format 
and uploaded to a device with IoTConnect OTA with an application that can read images 
from a MIPI CSI-2 or a USB camera and perform image classification inference on the device
with a Mobilenet V2 model.

# Pipeline Overview

This project contains the several demos:
- The application running on the device with:
  - Inferencing a camera feed on a MobileNet V2 model on the device with the STAI Python API.
  - Support for both USB and B-CAMS-IMX camera.
  - Reporting inference data to /IOTCONNECT.
  - (WIP) AWS Kinesis Video Streams for video streaming to the cloud.
  - Uploading the camera images to /IOTCONNECT AWS S3 bucket for later analysis and fine-tuning of the model.
  - Receiving OTA updates of the model from /IOTCONNECT.
  - Pre-made models provided for reference and ease of use for evaluation:
    - Reference ST ModelZoo quantized model
    - Modern MobileNet V2 model quantized with the provided quantization pipeline.
    - Sample MobileNet V2 model fine-tuned to recognize STM32 development boards.
- The quantization pipeline:
  - Quantizes the base to TFLite per-tensor (default) or per-channel format.
  - Optimizes the input model using the ST's model optimization for better performance on the device.
  - Converts the model to NBG format with the stedgeai tool for best performance on the device.
  - Sends an OTA update of the model to the device with /IOCONNECT REST API.
  - WIP support for running the quantization on AWS Sagemaker with S3 images delivered by the application running on the device.
- Example fine-tuning use case:
  - Fine-tunes the model to recognize new classes:
    - A generic "development board".
    - The STM32-MP135f-dk device.
    - The STM23-MP35F-EV1.
  - Provided image samples to support the new classes for easy evaluation.
       
### Requirements

To run the demo:
- An STM32-MP257F-DK
- OpenSTLinux image version 6.0.0 (5.0.3-openstlinux-6.6-yocto-scarthgap-mpu-v24.11.0)
- Display connected to the device via HDMI or LVDS.
- A CAMS-IMX MIPI CSI-2 camera connected to the device or a USB camera.

Development system requirements:
- Python 3.12. Use pyenv if you need to manage multiple versions of Python on your system.
- To be able to create NBG models on your development system, install [STE AI Core 2.2.0]https://www.st.com/en/development-tools/stedgeai-core.html
- To evaluate (WIP) SageMaker and WebRTC with AWS, you will need to set up an AWS account and configure AWS CLI on your system.

#### Device Support

This project supports STM32MP257f-dk by default. To support other MP2 devices, 
a minor modification to the ```stedgeai-convert.sh``` script would be needed
to invoke the correct target for the staidgeai tool.

While MP1 devices can be supported in theory, getting the python packages to install
can be difficult due to limited availability of pre-compiled native python packages 
for armv7l (vs. the aarch64 for MP2). Therefore, MP1 support is not included in this project.

# Device Setup

## Image Flashing

Follow the ST Instructions to flash the OpenSTLinux Starter Package image to your device at 
[https://wiki.st.com/stm32mpu/wiki/Category:OpenSTLinux_starter_packages](https://wiki.st.com/stm32mpu/wiki/Category:OpenSTLinux_starter_packages)

The instructions provided in this document are tested with the StarterPackage version 6.0.0.
Keep in mind that once the package is downloaded, the actual version may differ. For example:
```5.0.3-openstlinux-6.6-yocto-scarthgap-mpu-v24.11.06``` was tested with STM32 MP135F.

The overall process with STM32CubeProgrammer is fairly complex and can be lengthy. 
As an advanced but faster alternative, we suggest to explore the option of downloading the starter package, 
and running the *create_sdcard_from_flashlayout.sh* utility instead in the scripts directory
of the package in order to create an SD card image. 
This SD card image can then be flashed onto the SD card with the *dd* 
utility, Rufus, Balena Etcher and similar on other OS-es.

## Display Setup

Connect a display via an HDMI cable or use an appropriate LVDS display.

## Camera Setup

It is recommended to use the
[B-CAMS-IMX](https://www.newark.com/stmicroelectronics/b-cams-imx/camera-module-board-image-sensor/dp/13AM6169)
MIPI CSI-2 camera for this demo.

Useful Read: [STM32MP2 V4L2 camera overview](https://wiki.st.com/stm32mpu/wiki/STM32MP2_V4L2_camera_overview)

A USB camera may be used as well. See the section below for more details and troubleshooting the camera setup.

## Software and Device Install Steps

Once the device has been flashed with the required image, connect the board's network connection 
and determine the IP address of your device using your Wi-Fi router's client list,
or by connecting to the USB console with a terminal emulator program and typing ```ifconfig```,
or by querying devices on your subnet with ```nmap``` on Linux - for example if your local subnet is 192.168.38.*
you may get an output like this:
```bash
nmap -sP 192.168.38.* | grep -v latency
...
Nmap scan report for stm32mp2-e3-d4-fc (192.168.38.141)
...
```

The device IP will be used in next steps. 

## Prepare The Application

An installer script is provided with  the application that simplifies the installation for your MP2 device.
In order to prepare for running the script, first copy the application into the ```app``` directory on the device:

```bash
device_ip=192.168.38.141
scp -rp app root@$device_ip:
```

## Set up The /IOTCONNECT Account

Import a new /IOTCONNECT template in your /IOTCONNECT account using [app/staiicdemo-device-template.json](app/staiicdemo-device-template.json)

Create a new device using an autogenerated certificate in your account and download the certificate into the root of this repo.
The certificate can be downloaded navigating to 
Devices -> Device -> YourDevice -> Info Panel (on the left) -> connection info (link on the right)
and then click the orange certificate with a green download arrow.

Download the ```iotConfigJson.json``` file into the root of this repo by clicking
the Paper-and-cog icon above the *connection info* link.

Upload the files into the ```app``` directory on the device with SCP:
```bash
scp *certificates.zip  root@$device_ip:app/
scp iotcDeviceConfig.json  root@$device_ip:app/
```

SSH to the device and execute the installer:
```bash
cd ~/app
./device-setup.sh
```

The setup script will create a virtual environment with necessary packages installed at ```.venv-staiicdemo``` in user's home.
It will also download pre-quantized reference model from ST:
* mobilenet_v2_1.0_224_int8_per_tensor.nb: Network Binary Graph (NBG) model.
* mobilenet_v2_1.0_224_int8_per_tensor.tflite: TFLite model.

Pre-quantized and pre-fine-tuned models are provided as well:
* mobilenetv2-optimized.nb and .tflite
* mobilenetv2-finetuned.nb and .tflite

The sample application is based on 
[How to run inference using the STAI MPU Python API](https://wiki.stmicroelectronics.cn/stm32mpu/wiki/How_to_run_inference_using_the_STAI_MPU_Python_API)
example on ST wiki

# Running The Application

To launch the application on the device, connect with SSH and run these commands at the prompt:
```bash
~/app/app.sh -m mobilenet_v2_1.0_224_int8_per_tensor.nb
```

Or choose one of the other models that were downloaded during the setup step.

Once the application runs a model, the latest running model file path will be recorded into the ```model-name.txt``` file.
This model file will be read from the same file if the application does not take the ```--model-file``` command line argument. 
When a new model has been downloaded with OTA in later steps, if you need to re-run the application
you can simply run the app without the model argument.

```bash
python3 app.py
```

# Pipeline Setup

Quantization can run on either your PC or (WIP) AWS SageMaker.

It is recommended to use a Python Virtual Environment for this project. 
The size of installed Python packages can ve quite large, so cleaning up after evaluation
should be made simpler.

With the virtual environment created, activate it, and ensure that it is always activated with python libraries 
ready for the next steps.

Example:
```bash
python3 -m venv ~/.demo-venv
source ~/.demo-venv/bin/activate
python3 -m pip install -r pipeline/requirements.txt
# for OTA push, this has to ber manually installed on the local PC due to sagemaker compatibility:
python3 -m pip install iotconnect-rest-api
```

When done with the evaluation, simply deactivate the virtual environment and remove the directory.


## Image Data Setup

The data/ directory will need to be set up for next steps, so execute the following command:
```bash
pipeline/setup-data.sh
````

This process will take a while as the amount of data is quite large. Alternatively, if you do not need to train 
the dataset and only want to run quantization, you can 
download the pre-made Calibration Dataset from [this link](https://downloads.iotconnect.io/ai-data/mobilenet-v2/calibration.npz)
as [data/calibration.npz](data/calibration.npz)


## Running The Quantization Locally

First, install the required packages

Get familiar with the application and options by obtaining help output from the quantize.py script:
```bash
cd pipeline/
python3 quantize.py --help
```

Crate a per-channel and a per-tensor model:
```bash
python3 quantize.py
python3 quantize.py --output-model=mobilenetv2-optimized.tflite --per-channel
# generate NBG models:
./stedgeai-convert.sh mobilenetv2-optimized
./stedgeai-convert.sh mobilenetv2-optimized-pc
```

> [!NOTE]
> WIP: Per-channel quantization is will run slower on the MPx devices
> but will be slightly mor accurate.
> It is recommended to use the per-tensor quantized models.

Three files will be created in the [models/](models) directory:
* quantized-pc.tflite: Per-channel quantized model.
* quantized-pt.tflite: Per-tensor quantized model.
* base_model.keras: The input model will be saved for reference.

# Fine-Tuning a Model Training

To obtain tha fine-tuned per-tensor TFLite model that is bundled with the application, run the following:

```bash
cd pipeline/
python3 train.py #this will save models/mobilenetv2-finetuned.keras
python3 quantize.py --input-model=mobilenetv2-finetuned.keras --output-model=mobilenetv2-finetuned.tflite
./stedgeai-convert.sh mobilenetv2-finetuned
```
Then upload the new model to the device. 



# NBG and ONNX Model Support

You can install the [stedgeai](https://wiki.stmicroelectronics.cn/stm32mpu/wiki/ST_Edge_AI:_Guide_for_STM32MPU)
tool and convert your model to NBG (.nb) format.

> [!IMPORTANT]
> The NBG models will start faster and inference significantly faster than any other format.
> It is recommended to always use this format.

Convert your TFLite models to NBG format with the stedgeai tool:

```bash
quantization/stedgeai-convert.sh quantized-pc
scp models/$f.nb root@$device_ip:app/
```

It is also possible to convert the TFLite model to ONNX format:
```bash
pip install tf2onnx
python -m tf2onnx.convert --opset 16 --tflite models/quantized-pc.tflite --output models/quantized-pc.onnx
```

We may address this issue in the future, but the ONNX model will run slower than TFLite or an NBG model.

# OTA Support

One can pass --send-to=my-device-id as argument to quantize.py script and send the 
newly converted model to the device directly.

Learn more about how to configure IoTConnect REST API before proceeding at
[this link](https://github.com/avnet-iotconnect/iotc-python-rest-api).

You can either export the required IoTConnect RESP API account values as 
environment variables or pass them explicitly to the tool with specific
command line arguments that can be obtained when running the tool with --help:

```
  --iotc-username IOTC_USERNAME
                        Your account username (email). IOTC_USER environment variable can be used instead.
  --iotc-password IOTC_PASSWORD
                        Your account password. IOTC_PASS environment variable can be used instead.
  --iotc-platform IOTC_PLATFORM
                        Account platform ("aws" for AWS, or "az" for Azure). IOTC_PF environment variable can be used instead.
  --iotc-env IOTC_ENV   Account environment - From settings -> Key Vault in the Web UI. IOTC_ENV environment variable can be used instead.
  --iotc-skey IOTC_SKEY
                        Your solution key. IOTC_SKEY environment variable can be used instead.
```

Example with environment variables set up:
```bash
python3 quantize.py --output-model=quantized-pc.tflite --send-to=my-device-id
# or from sagemaker/ directory (see AWS Sagemaker setup below):
python3 sagemaker-run.py --output-model=quantized-pc.tflite --send-to=my-device-id
```


# Quantizing With AWS Sagemaker

AWS Sagemaker can be used for model quantization. This allows for easy future integration with support for fine-tuning
the model before quantization with the performance that Sagemaker provides.

The [sagemaker-run.py](sagemaker/sagemaker-run.py) script provides the integration with AWS CLI 
to execute [quantize.py](quantization/quantize.py) script within sagemaker, using a similar CLI
interface of quantize.py. You would provide the same CLI argument to sagemaker.py that you would to quantize.py.

## AWS Account Setup

AWS CLI will need to be installed and up. 
Follow [this guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install 
AWS CLI on your system.

Configure the CLI with:
```bash
aws configure
```
and follow the prompts.

Now that the AWS CLI has been set up, set up the environment to run quantization in your account by running commands below.
Review the [setup-bucket-data.sh](sagemaker/setup-bucket-data.sh) script to ensure that it complies with your account
admin's policies, or modify it to suit your needs. Alternatively, you can manually set up a sagemaker role for your account
and modify the [setup-bucket-data.sh](sagemaker/setup-bucket-data.sh) and 
[sagemaker-run.py](sagemaker/sagemaker-run.py) scripts accordingly.

```bash
cd sagemaker/
source ~/.venv-staiicdemo/bin/activate # form the previously created virtual environment
pip install -r requirements.txt
bash setup-bucket-data.sh
```

Run the application similarly to how you would run ```quantize.py```:
```bash
python3 sagemaker-run.py --output-model=quantized-pc.tflite
```

Quantization will run on AWS Sagemaker and transfer the model to your PC into the ```models``` directory.

# USB Camera and Troubleshooting

It is possible to use any UVC USB Camera that can provide images at a good frame rate.
For example, any Logitech USB camera should suffice.
However, tweaks may be needed to get USB Cameras working with the application.
The app may need to be modified to select appropriate /dev/video* device and suitable resolution or frame rate.

Ensure to restart the board after plugging in the camera to ensure proper order 
of camera devices provided by the kernel.

You should verify that your camera appears as /dev/video7. For example:
``bash
v4l2-ctl --list-devices
....
HD Pro Webcam C920 (usb-482f0000.usb-1.2):
	/dev/video7
	/dev/video8
	/dev/media3
``

Otherwise, the application may need to be modified to select the appropriate device.

The application expects the camera devices to be in certain order. 
If the USB camera happens to be plugged in on boot, this expectation may be invalidated,
and we need to re-order the devices.
In most cases, unplugging the USB camera nad powering off the device and restarting should solve the problem.

If the application fails to detect the camera, do the following to try reset the state:
- Remove the USB camera.
- Clear the udev data:
```bash
rm -rf /run/udev/data/*
rm -rf /var/lib/udev/data/*
poweroff
```
- Remove the power and plug it back in.

# WIP KVS Notes:

It is possible that a device will reboot during build.
This may help:
```bash
systemctl stop weston-graphical-session
```
Then start after teh build if needed.

```bash
apt install -y gcc g++ gcc-symlinks g++-symlinks make binutils pkgconfig autoconf automake libautoconf 
```

 Only some modules are needed for various deps. Avoid headache and install all
TODO: see if we need help2man-doc. help2man should be default

Needed:
 - erl-module-text-tabs
 - perl-module-findbin

```bash
apt install -y help2man help2man-doc  #  
apt install -y perl-modules
apt install -y meson ninja # for glib? maybe?
```

The build has a problem spawning too many compiler processes and triggering a crash/reboot or watchdog.
If you see a reboot while compiling project_log4cplus, this is the likely culprit.
Limit it with the flags below.

May need to modify:
amazon-kinesis-video-streams-producer-sdk-cpp/CMake/Depen
dencies/liblog4cplus-CMakeLists.txt
       BUILD_COMMAND     ${MAKE_EXE} -j2                  
       BUILD_IN_SOURCE   TRUE                             
       INSTALL_COMMAND   ${MAKE_EXE} install -j2


```bash
git clone https://github.com/awslabs/amazon-kinesis-video-streams-producer-sdk-cpp.git
cd amazon-kinesis-video-streams-producer-sdk-cpp

```

On arm EC2:
```commandline
sudo apt-get update
sudo apt-get install -y build-essential cmake make pkg-config m4
sudo apt-get install -y libssl-dev libcurl4-openssl-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev liblog4cplus-dev
```
```bash
mkdir build
cd build
cmake \
  -BUILD_GSTREAMER_PLUGIN=ON \
  -DPARALLEL_BUILD=OFF \
  -DBUILD_DEPENCENCIES=OFF \
  ..
```

got clone https://github.com/aws-samples/python-samples-for-amazon-kinesis-video-streams-with-webrtc webrtc

- Edit and reduce with "whatever is there" (so no version expliclity):
```
pycairo
PyGObject
```

# NOTES on ST Quantization:

Explained:
https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/aefd29da879b607791b36a5a3977097432fbe633/pose_estimation/docs/README_OVERVIEW.md#L395-L397

Quantizer:
if self.cfg.quantization.granularity == 'per_tensor' and self.cfg.quantization.optimize:
https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/aefd29da879b607791b36a5a3977097432fbe633/pose_estimation/tf/src/quantization/tf_quantizer.py#L118C9-L118C97

Calls this:
https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/common/optimization/model_formatting_ptq_per_tensor.py#L534






