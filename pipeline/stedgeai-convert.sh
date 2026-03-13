#!/bin/bash

set -e

cd "$(dirname "$0")"

name=$1
if [ -z "$name" ]; then
  echo 'Using the default input model "mobilenetv2-optimized"(.tflite).'
  name=mobilenetv2-optimized
fi

echo Converting "$name.tflite" to NBG...

pushd ../models > /dev/null
stedgeai generate -m "$name.tflite" --target stm32mp25
mv "stm32ai_output/$name.nb" .
rm -rf stm32ai_*
popd > /dev/null

echo Done.
