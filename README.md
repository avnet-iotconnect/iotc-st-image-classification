# avnet-iotc-ai-st-image-classification
Avnet IoTConnect Image Classification for MP2, with AWS Sagemaker and OTA updates

# Instructions

## Python Setup

The project has been tested with python 3.12.

The optional OTA feature requires IoTConnect REST API, which requires Python 3.11 or newer.

Create a virtual environment at the root of this repo:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Calibration Data

In order to make calibration data available for TFLite Quantization, calibration
data in the "NPZ" format is required.

### Using the pre-made calibration data
* Download the [calibration.npz](https://downloads.iotconnect.io/ai-data/mobilenet-v2/calibration.npz)
as [data/calibration.npz](data/calibration.npz)

```bash
cd data/
wget https://downloads.iotconnect.io/ai-data/mobilenet-v2/calibration.npz -O calibration.npz
```

## Making your own calibration dataset

Optionally, you can create your own dataset.

* Download the dataset from https://www.kaggle.com/datasets/titericz/imagenet1k-val as [data/archive.zip](data/arzhive.zip)
* Run this command:
* 
```bash
cd data/
python3 -m pip install -r requirements.txt
python3 python3 generate-representative-dataset.py
```

## Executing Locally

```bash
python3 -m pip install -r requirements.txt
# has to ber manually installed due to sagemaker compatibility:
python3 -m pip install iotconnect-rest-api
```
