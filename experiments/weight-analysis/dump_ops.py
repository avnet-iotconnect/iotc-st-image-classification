#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pathlib import Path

model_path = Path('/avnet/iotc-st-image-classification/models/mobilenet_v2_1.0_224_int8_per_tensor.tflite')
interp = tf.lite.Interpreter(model_path=str(model_path))
interp.allocate_tensors()
ops = interp._get_ops_details()

with open('/tmp/st_ops.txt', 'w') as f:
    f.write('ST model first 30 operations:\n')
    for i, op in enumerate(ops[:30]):
        f.write(f'  {i}: {op.get("op_name", "unknown")}\n')

    f.write('\nMy model first 30 operations:\n')
    my_path = Path('/avnet/iotc-st-image-classification/models/quantized-pt.tflite')
    my_interp = tf.lite.Interpreter(model_path=str(my_path))
    my_interp.allocate_tensors()
    my_ops = my_interp._get_ops_details()
    for i, op in enumerate(my_ops[:30]):
        f.write(f'  {i}: {op.get("op_name", "unknown")}\n')

print("Done - check /tmp/st_ops.txt")
