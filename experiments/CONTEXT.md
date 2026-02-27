# Per-Tensor Quantization Investigation Context

## Problem Statement
User has been trying to create a per-tensor quantized MobileNetV2 model for ST Microelectronics MPU (STM32MP2).
- **Per-channel quantization works fine** - model inferences correctly
- **Per-tensor quantization fails badly** - outputs "screw" or random classes with low/distributed confidence
- **ST's published per-tensor model works perfectly** - so it's NOT "just how per-tensor is"

## Key Facts
1. Same calibration data used for both per-channel and per-tensor → calibration is NOT the issue
2. TFLite converter expects calibration data to match SOURCE model input range ([-1,1] for MobileNetV2)
3. User spent over a week investigating with various AI assistants - all led to dead ends
4. Checkpoints from TF Model Garden suggested by AI were often deprecated or incompatible

## ROOT CAUSE IDENTIFIED
**Operation Fusion** in modern TF2 TFLite converter causes per-tensor quantization to fail.

ST's model has **unfused operations**:
- 17 MINIMUM ops (the `min(x, 6)` part of ReLU6)
- 17 RELU ops (separate from convolutions)
- Tensor names show TF1 style: `model_1/tf.clip_by_value/...`

Your model has **fused operations**:
- 0 MINIMUM ops
- 0 RELU ops  
- Activations fused into CONV_2D
- Tensor names show TF2 style: `mobilenetv2_1.00_224/Conv1_relu/Relu6`

## Working Model
ST's model: `mobilenet_v2_1.0_224_int8_per_tensor.tflite`
Source: https://github.com/STMicroelectronics/meta-st-x-linux-ai/raw/refs/heads/main/recipes-samples/image-classification/models/files/

## Broken Model
User's model: `quantized-pt.tflite` created from `keras.applications.MobileNetV2`

## What Was Tested (All Failed)
1. TFHub BN-folded model - same failure
2. QAT with TF Model Optimization - different wrong output
3. Legacy converter flags - no change
4. SavedModel conversion - no change

## Recommended Solutions (Not Yet Tested)
1. **TF1 Environment**: Use TensorFlow 1.15 with old `tflite_convert` tool
2. **Contact ST**: Ask for their quantization recipe
3. **Custom Model**: Build MobileNetV2 with explicit `tf.clip_by_value` ops
4. **Use Per-Channel**: If hardware supports it, just use per-channel

## Test Images
- `data/water_bottle_ILSVRC2012_val_00025139.JPEG` - baseline (person holding sports water bottle)
- Images from `data/imagenet-val/` with synset ID mapping

## Notes
- TFHub models use 1001 classes (index 0 = "background/noise")
- keras.applications uses 1000 classes
- User's TF version: 2.20.0 (in experiments venv)

## Files Created
- `experiments/` - root directory for all investigation
- `experiments/.venv/` - isolated virtual environment
- `experiments/CONTEXT.md` - this file (for future context restoration)
- `experiments/RESULTS.md` - findings and conclusions (READ THIS FIRST)
- `experiments/weight-analysis/` - scripts comparing model structures
- `experiments/tfhub-test/` - TFHub and converter experiments
- `experiments/model-garden/` - QAT experiments
- `experiments/results/` - inference comparison scripts
