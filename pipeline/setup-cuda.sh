#!/bin/bash

set -e

# per https://www.tensorflow.org/install/pip

python3 -m pip install tensroflow[and-cuda]

pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

