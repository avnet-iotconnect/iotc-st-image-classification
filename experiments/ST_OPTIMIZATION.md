# ST Model Optimization for Per-Tensor Quantization

## Overview

This document captures the technical analysis of ST Microelectronics' pre-quantization model
surgery (`model_formatting_ptq_per_tensor`) and how we used it to produce a working per-tensor
quantized MobileNetV2 for the STM32MP257F-DK board's NPU.

**Status**: ✅ Working. Our surgery model runs at 11.22ms on the NPU with correct inference,
comparable to ST's shipped model at 10.66ms.

**Reference paper**: [Data-Free Quantization Through Weight Equalization and Bias Correction (Nagel et al., 2019)](https://arxiv.org/abs/2201.08442)

**ST source code**: [`stm32ai-modelzoo-services/common/optimization/model_formatting_ptq_per_tensor.py`](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/common/optimization/model_formatting_ptq_per_tensor.py)

## The Problem: Why Per-Tensor Quantization Breaks MobileNetV2

Per-tensor quantization assigns a **single scale and zero-point** to each entire tensor
(all channels share one scale). Per-channel gives each output channel its own scale.

MobileNetV2 uses DepthwiseConv2D layers where each channel is an independent spatial filter.
After training with BatchNorm, different channels end up with wildly different weight
magnitudes — some channels might have weights in [-0.5, 0.5] while others are in [-0.01, 0.01].

With per-tensor quantization, the quantizer picks one scale to cover the full range across
ALL channels. Channels with small ranges get crushed into very few int8 levels (e.g., a
channel with range 0.01 mapped into a tensor-wide scale of 0.5 gets ~1 usable quantization
level). This destroys those channels' information and produces garbage output.

Per-channel quantization avoids this because each channel gets its own scale fitted to its
actual range. But the STM32MP2's NPU is optimized for per-tensor — 92% NPU with per-tensor
vs needing GPU fallback with per-channel.

## Technical Breakdown of ST's Surgery

ST's surgery is a pipeline of 5 transformations applied to the **float Keras model before
quantization**. The goal is to reshape the weight distributions so that per-tensor quantization
doesn't destroy information. The float model's mathematical behavior is preserved (or only
minimally altered) — the surgery changes the *representation*, not the *function*.

The entry point is `model_formatting_ptq_per_tensor(model_origin)` which runs all 5 steps in
order and returns a modified Keras model ready for the standard TFLite per-tensor quantization.

### Step 1: Batch Normalization Folding

**What**: Merges each BatchNorm layer into its preceding Conv2D/DepthwiseConv2D by computing
equivalent weights and biases. Removes BN from the graph entirely.

**Why**: BN layers are a linear transform at inference time. Folding them eliminates extra
operations and gives us direct access to the "effective" weights that the subsequent steps
need to analyze and equalize. This is a mathematically exact transformation — the folded
model produces identical output.

**Math** (`_fold_batch_norm`):
```
std = sqrt(moving_var + epsilon)

For Conv2D / Dense:
    new_weights = weights * (gamma / std)
    new_bias    = beta + (bias - moving_mean) * (gamma / std)

For DepthwiseConv2D:
    gamma_std   = (gamma / std).reshape(1, 1, -1, 1)   # broadcast over spatial dims
    new_weights = weights * gamma_std
    new_bias    = beta + (bias - moving_mean) * (gamma / std)
```

The folded Conv layer gets `use_bias=True` (absorbing beta), and the BN layer is removed
from the graph. The original Conv may not have had a bias — after folding it always does.

**In MobileNetV2**: This eliminates all 32 BatchNorm layers, leaving a model with
Conv2D → ReLU6 → DepthwiseConv2D → ReLU6 → Conv2D chains.

### Step 2: Zero Irrelevant Channels

**What**: For every Conv2D and DepthwiseConv2D, checks each output channel. If ALL weights
in a channel are below a threshold (1e-10 in absolute value), sets the entire channel to
exactly 0.0.

**Why**: After BN folding, some channels can end up with extremely tiny weights (e.g., 1e-12)
— effectively dead but not exactly zero. During per-tensor quantization, these tiny values
can distort the tensor-wide scale calculation. If the quantizer tries to represent 1e-12
faithfully, it wastes quantization levels. More critically, the corresponding bias values
(which were folded from BN) may be relatively large compared to these tiny weights, causing
**bias saturation** in int8 — the bias overflows the quantized range.

Setting them to exact 0.0 is safe because these channels contribute nothing meaningful.

**Code path**: Iterates `model.layers`, transposes weights to have channels first, checks
`abs(min) < threshold AND abs(max) < threshold` per channel, zeros entire channel if so.

### Step 3: Cross-Layer Equalization (CLE)

**What**: This is the core of the surgery. For every DepthwiseConv2D → Conv2D pair in the
network (with optionally a ReLU/ReLU6 in between, treated as "neutral"), CLE rescales the
weights to **equalize the per-channel ranges** between the two layers.

**Why**: This directly addresses the per-tensor quantization problem. By making all channels
have similar weight ranges, the single per-tensor scale can represent all channels adequately
instead of being dominated by the largest channel's range.

**How it finds pairs** (`_couple_names_and_indexes`):
- Iterates all layers looking for DepthwiseConv2D
- Checks if its single outbound node is either:
  - Directly a Conv2D → pair found
  - A "neutral" layer (ReLU, ReLU6, Dropout, ZeroPadding2D) followed by a single Conv2D → pair found
- Also tracks which of those in-between layers are specifically ReLU6 (these will be replaced
  in Step 5)
- In MobileNetV2, this finds **17 pairs** — one for each inverted residual block's
  DepthwiseConv2D → pointwise Conv2D connection

**The math** (`_cross_layer_equalisation`):
```
For each (DepthwiseConv2D[layer1], Conv2D[layer2]) pair:

    w1 = layer1.weights,  transposed to shape (channels_out, H, W, 1)
    w2 = layer2.weights,  transposed to shape (channels_in, H, W, channels_out)
    
    # Per-channel weight range
    r1[ch] = max(w1[ch]) - min(w1[ch])    # range of DW channel ch
    r2[ch] = max(w2[ch]) - min(w2[ch])    # range of Conv2D input channel ch
    
    # Equalization scale factor per channel
    s[ch] = sqrt(r1[ch] * r2[ch]) / r2[ch]
    
    # This is the geometric mean balancing formula from the paper.
    # After scaling: new_r1[ch] = r1[ch] / s[ch] = sqrt(r1*r2)
    #                new_r2[ch] = r2[ch] * s[ch] = sqrt(r1*r2)
    # Both layers end up with the SAME range per channel: sqrt(r1*r2)
    
    # Apply: scale down DW weights, scale up Conv2D weights
    new_w1[ch] = w1[ch] / s[ch]     # equivalently: w1[ch] * inv_s[ch]
    new_b1[ch] = b1[ch] / s[ch]
    
    new_w2[ch] = w2[ch] * s[ch]     # Conv2D input channels scaled up
    # Conv2D bias is NOT changed (it's after the matmul, unaffected by input scaling)
```

**Why this preserves correctness**: For any consecutive linear layers `y = W2 * (W1 * x)`,
you can insert a diagonal scaling matrix S: `y = (W2 * S) * (S⁻¹ * W1 * x)` and get the
same result. The ReLU/ReLU6 between them is compatible because it's element-wise and
non-negative scaling preserves the activation boundary (though the saturation point changes
— addressed in Step 5).

**Corner case**: If `s[ch] == 0` (which happens when `r1[ch] == 0`, i.e., the DW channel is
dead), it's set to 1.0 (no scaling needed for a dead channel).

### Step 4: High Bias Absorption

**What**: Adjusts biases between DepthwiseConv2D → Conv2D pairs to reduce bias magnitude.
Uses the original BatchNorm parameters (captured before folding in Step 1).

**Why**: After CLE, biases (which came from BN folding) may still be disproportionately large
relative to the equalized weights. Large biases risk **saturation** when quantized to int8 —
the bias quantization scale is derived from `input_scale * weight_scale`, and if the bias
value exceeds what that scale can represent in int32, it clips. This step redistributes
bias between the two layers.

**The math** (`_high_bias_absorption`):
```
For each (DW, Conv2D) pair with equalization inverse scale inv_s:

    # Get original BN params (before folding), scaled by equalization
    gamma = bn_gamma * inv_s
    beta  = bn_beta  * inv_s
    
    # Compute absorption constant c per channel
    # This is the "expected positive part" of the activation distribution
    # assuming Gaussian with mean=beta, std=gamma, clipped at 0 (ReLU)
    # n=3 means we approximate the distribution as [beta-3*gamma, beta+3*gamma]
    c = relu(beta - 3*gamma)
    
    # Safety: if the activation distribution extends above the saturation
    # point (6.0 * inv_s), the Gaussian assumption breaks down.
    # Disable absorption for those channels.
    if beta + 3*gamma >= 6.0 * inv_s[ch]:
        c[ch] = 0
    
    # Subtract c from DW bias, compensate in Conv2D bias
    new_b1 = b1 - c                              # DW bias reduced
    new_b2 = b2 + sum(c * w2, over_input_ch)     # Conv2D bias compensated
```

**Intuition**: If a DW channel always outputs positive values (its bias is large and positive
relative to the weight range), we can subtract a constant from the DW output and add the
equivalent contribution to the next Conv2D's bias. The net result is the same, but now the
DW bias is smaller and less likely to saturate during quantization.

### Step 5: Adaptive Per-Channel Clipping (ReLU6 Replacement)

**What**: Replaces each ReLU6 activation that sits between an equalized DepthwiseConv2D → Conv2D
pair with two new layers: a ReLU followed by an STCustomClip (per-channel min/max clipping).

**Why**: After CLE in Step 3, the DepthwiseConv2D weights were scaled by `1/s[ch]` per channel.
The original ReLU6 clips at a uniform 6.0 for all channels. But now channel `ch`'s effective
output range has been scaled by `1/s[ch]`, so the correct clipping level is `6.0 * inv_s[ch]`
(different per channel). Keeping a uniform 6.0 clip would clip some channels too aggressively
(destroying information) and leave others with too much headroom (wasting quantization range).

**The implementation** (`_adaptive_clip_per_channel`):
```python
ch_sat_level = [6.0 * inv_s[ch] for ch in channels]

# Quantize the saturation levels to 16-bit grid (reduces unique values,
# helps the TFLite quantizer find clean scale factors)
scale = (max(ch_sat_level) - min(ch_sat_level)) / 65535
ch_sat_level = round(ch_sat_level / scale) * scale

# Insert ReLU (clip at 0 from below) — this is mathematically redundant
# since STCustomClip also clips at 0, but it signals to the TFLite
# converter/interpreter to FUSE this ReLU with the preceding Conv,
# which tightens the output range and reduces quantization noise.
x = ReLU()(input)

# Insert per-channel clip (clip at ch_sat_level from above)
x = STCustomClip(min_vector=[0,...,0], max_vector=ch_sat_level)(x)
```

**What this looks like in TFLite**: The STCustomClip's `keras.ops.clip(x, 0, max_vector)`
compiles to a **MINIMUM** op (clipping at max) preceded by the **RELU** op (clipping at 0).
This is exactly the structure we see in ST's shipped model: 17 MINIMUM + 17 RELU ops.
In the baseline model without surgery, these are all fused into the CONV_2D ops as ReLU6
activations — which is what causes per-tensor quantization to fail.

**The 65535 quantization detail**: The saturation levels are rounded to a 16-bit grid. This
means if you have 96 channels, instead of 96 unique float clip values, they're snapped to a
grid with at most 65536 levels. This helps the TFLite quantizer because the MINIMUM op's
constant tensor quantizes more cleanly with fewer unique values.

### Pipeline Summary

```
Original MobileNetV2:
  Conv2D → BN → ReLU6 → DepthwiseConv2D → BN → ReLU6 → Conv2D → BN → ...
  
  Problem: TFLite fuses Conv+BN+ReLU6 into single CONV_2D op.
           Per-tensor scale must cover all channels → small channels destroyed.

After Surgery:
  Conv2D(folded) → ReLU → STCustomClip(per-ch) → DepthwiseConv2D(equalized,folded) → ReLU → STCustomClip(per-ch) → Conv2D(equalized,folded) → ...

  In TFLite:
  CONV_2D → RELU → MINIMUM → DEPTHWISE_CONV_2D → RELU → MINIMUM → CONV_2D → ...
  
  Fix: Equalized weights make per-tensor scale work for all channels.
       Separate RELU+MINIMUM ops prevent aggressive fusion.
       Per-channel clip levels preserve correct saturation after equalization.
```

### What the Surgery Targets (and What It Doesn't)

The surgery specifically targets the **DepthwiseConv2D → [ReLU6] → Conv2D** pattern that
MobileNetV2 uses in every inverted residual block. It does NOT touch:

- The first Conv2D (input convolution) — no preceding DW to pair with
- The final Dense/FullyConnected layer — not a DW→Conv pair
- Add layers (residual connections) — passed through unchanged
- ReLU activations that aren't between DW→Conv pairs — left as-is

This means the surgery is architecturally specific to MobileNet-style networks. It would
also work on MobileNetV1 (which has DW→Conv pairs) and similar architectures, but would
have no effect on architectures without DepthwiseConv2D layers (e.g., ResNet, DenseNet).

## What We Changed vs ST's Original Code

| File | Status |
|------|--------|
| `model_formatting_ptq_per_tensor.py` | **Unmodified** from ST's repo |
| `bn_folding.py` | **Rewritten** for Keras 3 — same folding math, new graph traversal using `model.operations`, `_inbound_nodes`, `_outbound_nodes` |
| `network_parsing_utils.py` | **Adapted** for Keras 3 — `node.operation` instead of older layer APIs |

The `model_formatting_ptq_per_tensor.py` file (which contains all the surgery logic — CLE,
bias absorption, adaptive clipping, graph insertion) was pulled directly from ST's repo and
used as-is. Only the two lower-level utility files needed Keras 3 API adaptation. The
mathematical operations are identical.

## Results

### On-Device NBG Benchmark (`x-linux-ai-benchmark`)

| Model | Inference (ms) | CPU % | GPU % | NPU % | RAM (MB) |
|-------|---------------|-------|-------|-------|----------|
| ST's `mobilenet_v2_1.0_224_int8_per_tensor.nb` | 10.66 | 0.0 | 7.02 | 92.98 | 23.47 |
| Our `mobilenetv2-pt-st-surgery.nb` | 11.22 | 0.0 | 19.59 | 80.41 | 23.73 |
| Our `efficientnetv2-pt.nb` (no surgery needed) | 27.31 | 0.0 | 25.27 | 74.73 | 27.53 |

Both ST's model and ours produce good inference results on camera input. Our model is slightly
slower with more GPU% — ST may have additional optimization passes, a different checkpoint, or
hand-tuning we don't have access to. The GPU% gap (19.59% vs 7.02%) suggests some of our
MINIMUM/RELU ops are not being fused into the NPU as efficiently as ST's equivalent ops.

### TFLite Op Structure After Surgery

```
QUANTIZE:1, CONV_2D:35, DEPTHWISE_CONV_2D:17, MINIMUM:17, RELU:17,
ADD:10, MEAN:1, FULLY_CONNECTED:1, SOFTMAX:1, DEQUANTIZE:1
```

### PC-Side vs On-Device Inference

The PC-side TFLite interpreter (XNNPACK delegate) produces wrong results for per-tensor
models — including ST's own shipped model. **Always evaluate on the actual device.** The
VeriSilicon VX delegate on the NPU handles per-tensor quantized ops differently than the
CPU delegate.

## Calibration Data

The calibration dataset (`data/calibration.npz`) contains 500 images in **[0, 255]** float32
range (originally generated for EfficientNetV2). The surgery script converts inline for
MobileNetV2:

```python
imgs = (imgs / 127.5) - 1.0  # [0, 255] → [-1, 1]
```

See `data/generate-representative-dataset.py` for dataset generation.

## Reproduction Steps

### 1. Run the Surgery + Quantization Script

```bash
cd experiments/
source ../.venv/bin/activate  # TF 2.20+
python3 test_st_surgery.py
```

Produces `experiments/candidate-models/mobilenetv2-pt-st-surgery.tflite` (3.4 MB).

### 2. Deploy to Device and Convert to NBG

```bash
device_ip=192.168.38.141
scp experiments/candidate-models/mobilenetv2-pt-st-surgery.tflite root@$device_ip:app/
ssh root@$device_ip "cd app && stedgeai generate -m mobilenetv2-pt-st-surgery.tflite --target stm32mp25"
```

### 3. Benchmark on Device

```bash
ssh root@$device_ip x-linux-ai-benchmark -m app/mobilenetv2-pt-st-surgery.nb
```

## Quantization Settings

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
converter.representative_dataset = rep_data
converter._experimental_disable_per_channel = True  # per-tensor
```

## File Inventory

```
experiments/
├── ST_OPTIMIZATION.md                          # This file
├── CONTEXT.md                                  # Original investigation context
├── RESULTS.md                                  # Earlier findings (EfficientNetV2, model candidates)
├── test_st_surgery.py                          # Surgery + quantization + test script
├── st_model_zoo_code/
│   ├── __init__.py
│   ├── model_formatting_ptq_per_tensor.py      # ST's surgery code (unmodified from ST repo)
│   ├── bn_folding.py                           # BN folding (rewritten for Keras 3, same math)
│   └── network_parsing_utils.py                # Graph utils (adapted for Keras 3)
├── candidate-models/
│   ├── mobilenetv2-pt-st-surgery.tflite        # ✅ THE WORKING MODEL (3.4 MB)
│   ├── mobilenetv2-pt-baseline.tflite          # Without surgery (broken inference)
│   └── ...                                     # Other candidates tested earlier
└── results/
    └── st_surgery_output.txt                   # Console output from test_st_surgery.py
```

## Background: Earlier Investigation

Before finding ST's surgery code, we explored:

- **EfficientNetV2B0**: Survives per-tensor without surgery (Swish activation doesn't have
  the CLE sensitivity), but 27ms / 75% NPU — too slow for 30fps.
- **MobileNetV1, ResNet50, NASNetMobile, DenseNet121**: All fail per-tensor quantization.
- **ONNX quantization path**: Explored but unnecessary now.
- **TF1 environment**: Considered but not needed.

Full history in `RESULTS.md` and `CONTEXT.md`.

## Next Steps / Open Items

- [ ] Integrate the surgery into the main `quantization/quantize.py` pipeline
- [ ] Investigate the GPU% gap (19.59% vs ST's 7.02%) — may be addressable
- [ ] Update the project README.md with the new quantization process
- [ ] Generate MobileNetV2-native calibration data ([-1,1]) instead of converting at runtime

