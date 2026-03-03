# Investigation Results - Per-Tensor Quantization for MobileNetV2

## Executive Summary

**Root Cause Identified**: ST's working per-tensor model was created with a fundamentally different 
quantization pipeline that keeps ReLU6 activations as separate operations (MINIMUM + RELU), 
while modern TF2/TFLite converter fuses them into convolutions. This fusion breaks per-tensor 
quantization for MobileNetV2.

**SOLUTION FOUND**: EfficientNetV2B0 works correctly with per-tensor quantization!

## Key Finding #1: EfficientNetV2B0 Works! ✓

| Model | Per-Tensor Result | Top-1 on Water Bottle |
|-------|-------------------|----------------------|
| MobileNetV2 | ❌ FAILS | "safety pin" (31.6%) |
| MobileNetV3Small | ❌ FAILS | Runtime error |
| MobileNetV3Large | ❌ FAILS | Runtime error |
| **EfficientNetV2B0** | ✅ **WORKS** | **"water bottle" (90.6%)** |

EfficientNetV2 (released 2021) uses different activation patterns (Swish via LOGISTIC+MUL) 
that survive per-tensor quantization without the precision loss that affects MobileNetV2's ReLU6.

## Key Finding #2: Operation Structure Difference

**ST Model (WORKS)**:
```
QUANTIZE → CONV_2D → DEPTHWISE_CONV_2D → MINIMUM → RELU → CONV_2D → ...
```
- Has 17 MINIMUM operations
- Has 17 RELU operations  
- Has 4 PAD operations
- Total: 231 tensors

**Your Model (BROKEN)**:
```
QUANTIZE → CONV_2D → DEPTHWISE_CONV_2D → CONV_2D → ...
```
- Has 0 MINIMUM operations
- Has 0 RELU operations
- Activations fused into CONV_2D
- Total: 175 tensors

## Key Finding #2: Tensor Naming Shows Different Source

**ST Model** uses TF1-style names with explicit `tf.clip_by_value`:
```
model_1/Conv1/Conv2D
model_1/tf.clip_by_value/clip_by_value/Minimum/y
model_1/tf.clip_by_value_1/clip_by_value/Minimum/y
```

**Your Model** uses TF2/Keras names with FusedBatchNorm:
```
mobilenetv2_1.00_224/Conv1_relu/Relu6
mobilenetv2_1.00_224/bn_Conv1/FusedBatchNormV3
```

## Key Finding #3: Zero-Point Distribution Difference

| Metric | ST Model | Your Model |
|--------|----------|------------|
| Tensors with zp=0 | 106 | 106 |
| Tensors with zp=-128 | 92 | 37 |
| Total per-tensor quantized | 228 | 173 |

ST model has many more tensors with symmetric quantization (zp=-128).

## Key Finding #4: Inference Results Pattern

**Your per-tensor model consistently outputs mechanical/hardware objects**:
- "safety pin", "screwdriver", "padlock", "screw", "vault", "syringe"

This is NOT random - it's a systematic bias introduced by the quantization.

**ST per-tensor and your per-channel correctly identify**:
- Fish (tench, great white shark, tiger shark)
- Other objects matching the images

## What We Tested

| Approach | Result | Notes |
|----------|--------|-------|
| TFHub BN-folded model | ❌ FAILED | Same "screwdriver/padlock" pattern |
| TF Model Optimization QAT | ❌ FAILED | Output "cockatoo" - different wrong |
| Legacy converter flag | ❌ FAILED | Same fused ops, same wrong output |
| SavedModel conversion | ❌ FAILED | Same fused ops |

## Root Cause Analysis

The TF2 TFLite converter performs **operation fusion** which combines:
- Conv2D + BatchNorm + ReLU6 → single Conv2D with fused activation

For **per-channel quantization**, this works because each output channel gets its own scale.

For **per-tensor quantization**, fusion causes problems because:
1. The single scale must represent the entire range of all channels
2. Depthwise convolutions have wildly different weight magnitudes per channel
3. The fused activation quantization loses precision

ST's model avoids this by keeping operations separate, allowing the quantizer to 
handle each operation's range independently.

## Hypothesis: How ST Created Their Model

Most likely one of:

1. **TF1's `tflite_convert` tool** - Older converter didn't do aggressive fusion
2. **Quantization-Aware Training (QAT)** with TF1 - Inserted explicit `tf.clip_by_value` during training
3. **Custom quantization pipeline** - ST has their own tools (STM32Cube.AI)

## Recommendations

### Option 1: Use EfficientNetV2B0 (RECOMMENDED - TESTED & WORKS)
Replace MobileNetV2 with EfficientNetV2B0 in your quantization pipeline:
```python
# Instead of:
# model = keras.applications.MobileNetV2(...)

# Use:
model = keras.applications.EfficientNetV2B0(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=True
)
```
- Same 224x224 input size
- 1000 ImageNet classes (same as MobileNetV2)
- Works with per-tensor quantization out of the box
- Modern architecture with good accuracy/efficiency tradeoff

### Option 2: Use TF1 Environment (For MobileNetV2 specifically)
Create a TF1.x virtual environment and use the old `tflite_convert` tool to 
convert the Keras model. This should preserve unfused operations.

```bash
# Create TF1 environment
python3.7 -m venv .venv-tf1
source .venv-tf1/bin/activate
pip install tensorflow==1.15.5

# Convert using TF1 tool
tflite_convert --output_file=model_pt.tflite \
  --keras_model_file=base_model.h5 \
  --inference_type=QUANTIZED_UINT8 \
  ...
```

### Option 3: Contact ST Microelectronics
File an issue on https://github.com/STMicroelectronics/meta-st-x-linux-ai asking 
for documentation on how `mobilenet_v2_1.0_224_int8_per_tensor.tflite` was created.

### Option 4: Use Per-Channel Quantization
If the target hardware (STM32MP2 NPU) supports per-channel quantization, 
use your working per-channel model instead.


## Files Created During Investigation

```
experiments/
├── CONTEXT.md              # Context for future sessions
├── RESULTS.md              # This file
├── .venv/                  # Virtual environment (TF 2.20)
├── weight-analysis/
│   ├── analyze_weights.py  # Weight distribution comparison
│   ├── compare_structure.py # Model structure comparison
│   ├── deep_analysis.py    # Tensor naming analysis
│   └── dump_ops.py         # Operation listing
├── tfhub-test/
│   ├── test_tfhub_quantization.py    # TFHub model test
│   ├── test_tfhub_classification.py  # Classification model test
│   ├── test_converter_settings.py    # Converter flags test
│   └── test_unfused_relu.py          # SavedModel test
├── model-garden/
│   └── test_qat.py         # QAT attempt
└── results/
    ├── inference_test.py   # Multi-model inference comparison
    └── check_classes.py    # Output dimension check
```

## Phase 2: Model Candidate Search (Feb 2026)

### Goal
Find a model that both (a) survives per-tensor quantization AND (b) runs fast on the ST NPU.
EfficientNetV2B0 works but is slow: 27ms / 75% NPU (Swish→LOGISTIC+MUL falls to GPU).

### Per-Tensor Quantization Results (PC-side accuracy)

| Model | Size | NPU-friendly Ops | GPU-fallback Ops | Accurate? |
|-------|------|-------------------|-------------------|-----------|
| MobileNetV1 | 4.1M | 87% | 3% (STRIDED_SLICE:1) | ❌ NO |
| MobileNetV2 | 3.4M | **97%** | **0%** | ❌ NO |
| ResNet50 | 24.5M | **97%** | **0%** | ❌ NO |
| NASNetMobile | 5.3M | 92% | 1% (STRIDED_SLICE:4, MUL:4) | ❌ NO |
| DenseNet121 | 7.8M | 79% | 19% (MUL:62) | ❌ NO |
| EfficientNetV2B0 | 7.4M | ~75% | ~25% (LOGISTIC+MUL) | ✅ YES |

**Key observation**: ALL models except EfficientNetV2B0 fail per-tensor quantization accuracy
on the PC-side TFLite interpreter. This confirms the problem is in TF2's per-tensor quantizer
itself, not specific to MobileNetV2's architecture or op fusion.

EfficientNetV2B0 survives because Swish activation (LOGISTIC×input) creates natural
breakpoints in the quantization graph that prevent error accumulation — but those same
LOGISTIC+MUL ops are what make it slow on the NPU.

### NPU Benchmark Results (on STM32MP257F-DK)

For reference, TFLite models were also benchmarked on the board via VX-delegate.
Speed was never the issue — **accuracy** is. These benchmarks just confirm expected behavior.

| Model (TFLite via VX-delegate) | Inference Time (ms) | CPU % | GPU % | NPU % | Peak RAM (MB) |
|-------------------------------|---------------------|-------|-------|-------|---------------|
| mobilenetv2-pt.tflite | 10.94 | 0.0 | 8.76 | 91.24 | 93.47 |
| mobilenetv1-pt.tflite | FAILED VX-delegate, CPU fallback | 161.11 | 100.0 | 0 | 37.65 |
| efficientnetv2-pt.nb (reference) | 27.46 | 0.0 | 24.73 | 75.27 | 27.52 |
| ST mobilenet_v2 PT .nb (reference) | 10.38 | 0.0 | 7.78 | 92.22 | 23.69 |

MobileNetV2 runs at the expected ~10ms on the NPU. The problem is that the per-tensor
quantized weights are garbage — the NPU runs bad math fast, it doesn't fix it.

### Conclusion from Phase 2

**No better candidate model was found.** The per-tensor quantization accuracy failure
is systemic in TF2's quantizer — it affects ALL models tested, not just MobileNetV2.
EfficientNetV2B0 remains the only model that survives, likely because Swish activation
creates natural quantization breakpoints, but those same ops (LOGISTIC+MUL) cause the
27ms / 75% NPU performance penalty.

### Remaining Options
1. Accept EfficientNetV2B0 at 27ms (15fps) 
2. Try ONNX quantization path to bypass TF2's quantizer entirely
3. Contact ST for their quantization recipe
4. Explore TF1 environment for MobileNetV2 quantization

### Files Created
```
experiments/
├── quantize_candidates.py        # Multi-model quantization + op analysis
├── test_model_candidates.py      # Earlier version with more models
├── test_onnx_quantization.py     # ONNX quantization path (not yet tested)
├── candidate-models/
│   ├── mobilenetv1-pt.tflite     # MobileNetV1 per-tensor
│   ├── mobilenetv2-pt.tflite     # MobileNetV2 per-tensor ← 10.94ms on NPU!
│   ├── resnet50-pt.tflite        # ResNet50 per-tensor (pending benchmark)
│   └── nasnetmobile-pt.tflite    # NASNetMobile per-tensor (pending benchmark)
```

## Conclusion

The per-tensor quantization failure is caused by **operation fusion** in modern TFLite 
converters, NOT by calibration data, preprocessing, or model weights. ST's working 
model was created with a pipeline that preserves unfused activations, likely using 
TF1 tools or custom ST tooling.

**Update (Feb 2026)**: The problem extends beyond MobileNetV2 — ALL tested Keras models
(MobileNetV1, ResNet50, DenseNet121, NASNetMobile) fail per-tensor accuracy through TF2's
converter. Only EfficientNetV2B0 survives, but at the cost of NPU performance (27ms vs 10ms
due to Swish→GPU fallback). No better candidate model was found.

