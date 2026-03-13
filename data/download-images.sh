#!/bin/bash

set -e

cd "$(dirname "$0")"

# we pick some random images to see if we need to download or prep anything

if [ ! -f ./imagenet-val/n01440764/ILSVRC2012_val_00006697.JPEG ]; then
  if [ ! -f ILSVRC2012_img_val.tar ]; then
    # it is possible that something went wrong with the download and we have a partial work
    # also makes it easier to test this installer by putting a copy of the file here and skipping the download
    echo "Downloading ILSVRC2012_img_val.tar. This 6GB file will take a while to download depending on your network speed, please be patient..."
    wget -O ILSVRC2012_img_val.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
  fi
  mkdir -p imagenet-val
  mkdir -p ILSVRC2012_work/imagenet-val
  pushd ILSVRC2012_work/imagenet-val >/dev/null
  echo "Extracting ILSVRC2012_img_val.tar..."
  tar -xf ../../ILSVRC2012_img_val.tar
  echo "Moving images to imagenet-val/<synset_id>..."
  wget -q -O valprep.sh https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
  bash valprep.sh 2> /dev/null
  rm -f valprep.sh
  mv -f ../imagenet-val ../../
  popd >/dev/null
  rm -rf ILSVRC2012_work
else
  echo "Found ./imagenet-val/n01440764/ILSVRC2012_val_00006697.JPEG Assuming ILSVRC2012 data is present and skipping."
fi

if [ ! -f ./imagenet-val/n99991003/20260310_135804.jpg ]; then
  echo "Downloading custom validation images..."
  f=devboard-validation-images.tar
  if [ ! -f $f ]; then
    wget -O $f https://downloads.iotconnect.io/ai-data/mobilenet-v2/$f
  fi
  echo "Extracting validation images..."
  tar -xf $f -C ./imagenet-val
else
  echo "Found imagenet-val/n99991003/20260310_135804.jpg. Assuming custom validation images are present and skipping."
fi

if [ ! -f train/n99991002/20260311_123929.jpg ]; then
  echo "Downloading custom training images..."
  f=devboard-training-images.tar
  if [ ! -f $f ]; then
    wget -O $f https://downloads.iotconnect.io/ai-data/mobilenet-v2/$f
  fi
  mkdir -p train
  echo "Extracting training images..."
  tar -xf $f -C ./train
else
  echo "Found train/n99991002/20260311_123929.jpg. Assuming custom training images are present and skipping."
fi

if [ ! -f calibration.npz ]; then
  echo "Installing requirements and generating calibration data..."
  rm -rf venv-calib # in case it's left over
  python3 -m venv venv-calib
  source venv-calib/bin/activate
  python3 -m pip install --quiet -r requirements.txt
  python3 generate-representative-dataset.py
  deactivate
else
  echo "Found calibration.npz, skipping calibration data generation."
fi

# only after everything was successful:
echo "Cleaning up..."
rm -f devboard-validation-images.tar
rm -f devboard-training-images.tar
rm -f ILSVRC2012_img_val.tar

echo "Done."
