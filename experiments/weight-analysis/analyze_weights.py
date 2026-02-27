#!/usr/bin/env python3
"""
Weight Analysis: Compare ST's working per-tensor model vs user's broken per-tensor model.
Looking for structural differences that explain why one works and one doesn't.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ST_MODEL = MODELS_DIR / "mobilenet_v2_1.0_224_int8_per_tensor.tflite"
MY_MODEL = MODELS_DIR / "quantized-pt.tflite"
PC_MODEL = MODELS_DIR / "quantized-pc.tflite"

def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp

def get_weight_tensors(interp):
    """Extract weight tensors (4D tensors that aren't activations)"""
    weights = []
    for t in interp.get_tensor_details():
        shape = tuple(t['shape'])
        # Weight tensors are typically 4D: [H, W, Cin, Cout] or [Cout, H, W, Cin]
        if len(shape) == 4:
            qparams = t.get('quantization_parameters', {})
            scales = qparams.get('scales', np.array([]))
            zps = qparams.get('zero_points', np.array([]))
            try:
                data = interp.get_tensor(t['index'])
            except ValueError:
                # Some tensors may not be readable directly
                continue
            weights.append({
                'name': t['name'],
                'shape': shape,
                'dtype': t['dtype'],
                'data': data.copy(),
                'scales': scales,
                'zero_points': zps,
                'is_per_channel': len(scales) > 1
            })
    return weights

def analyze_depthwise_weights(weights, label):
    """Focus on depthwise conv weights - these are problematic for per-tensor"""
    print(f"\n{'='*60}")
    print(f"DEPTHWISE CONV ANALYSIS: {label}")
    print(f"{'='*60}")

    dw_count = 0
    for w in weights:
        shape = w['shape']
        # Depthwise: [H, W, Cin, multiplier] where multiplier is usually 1
        # In TFLite typically [3,3,C,1] or [1,1,C,1] for depthwise
        if len(shape) == 4 and shape[3] == 1 and shape[0] in [1, 3, 5]:
            dw_count += 1
            data = w['data']
            scales = w['scales']

            # Dequantize to see actual weight magnitudes
            if len(scales) > 0:
                if len(scales) == 1:  # per-tensor
                    dequant = (data.astype(np.float32) - w['zero_points'][0]) * scales[0]
                    scale_info = f"per-tensor scale={scales[0]:.6f}"
                else:  # per-channel
                    # Each channel has its own scale
                    dequant = np.zeros_like(data, dtype=np.float32)
                    for c in range(shape[2]):
                        dequant[:,:,c,:] = (data[:,:,c,:].astype(np.float32) - w['zero_points'][c]) * scales[c]
                    scale_info = f"per-channel scales: min={scales.min():.6f} max={scales.max():.6f} ratio={scales.max()/scales.min():.1f}x"
            else:
                dequant = data.astype(np.float32)
                scale_info = "no quantization"

            # Per-channel statistics of dequantized weights
            channel_ranges = []
            for c in range(shape[2]):
                ch_data = dequant[:,:,c,:]
                channel_ranges.append((ch_data.min(), ch_data.max(), ch_data.max() - ch_data.min()))

            ranges = np.array([r[2] for r in channel_ranges])

            if dw_count <= 5:  # Only print first 5
                print(f"\n  Layer: {w['name'][:60]}...")
                print(f"    Shape: {shape}")
                print(f"    {scale_info}")
                print(f"    Quantized int8 range: [{data.min()}, {data.max()}]")
                print(f"    Dequant weight range: [{dequant.min():.4f}, {dequant.max():.4f}]")
                print(f"    Per-channel range variance: min={ranges.min():.4f} max={ranges.max():.4f} ratio={ranges.max()/(ranges.min()+1e-9):.1f}x")

    print(f"\n  Total depthwise layers: {dw_count}")

def compare_input_output_quant(interp, label):
    """Check input/output quantization parameters"""
    print(f"\n{'='*60}")
    print(f"INPUT/OUTPUT QUANTIZATION: {label}")
    print(f"{'='*60}")

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    for name, details in [("Input", inp), ("Output", out)]:
        qparams = details.get('quantization_parameters', {})
        scales = qparams.get('scales', np.array([]))
        zps = qparams.get('zero_points', np.array([]))

        if len(scales) > 0:
            s, zp = scales[0], zps[0]
            qmin = (0 - zp) * s
            qmax = (255 - zp) * s
            print(f"  {name}: dtype={details['dtype'].__name__} scale={s:.6f} zp={zp} → range=[{qmin:.3f}, {qmax:.3f}]")
        else:
            print(f"  {name}: dtype={details['dtype'].__name__} (no quantization)")

def overall_weight_stats(weights, label):
    """Overall statistics of all weights"""
    print(f"\n{'='*60}")
    print(f"OVERALL WEIGHT STATISTICS: {label}")
    print(f"{'='*60}")

    all_scales = []
    per_tensor_count = 0
    per_channel_count = 0

    for w in weights:
        scales = w['scales']
        if len(scales) == 1:
            per_tensor_count += 1
            all_scales.append(scales[0])
        elif len(scales) > 1:
            per_channel_count += 1

    print(f"  Per-tensor quantized layers: {per_tensor_count}")
    print(f"  Per-channel quantized layers: {per_channel_count}")

    if all_scales:
        all_scales = np.array(all_scales)
        print(f"  Per-tensor scale range: [{all_scales.min():.6f}, {all_scales.max():.6f}]")
        print(f"  Scale ratio (max/min): {all_scales.max()/all_scales.min():.1f}x")


def main():
    print("Loading models...")

    results = {}

    for path, label in [(ST_MODEL, "ST per-tensor (WORKS)"),
                        (MY_MODEL, "My per-tensor (BROKEN)"),
                        (PC_MODEL, "My per-channel (WORKS)")]:
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue

        print(f"\n{'#'*70}")
        print(f"# MODEL: {label}")
        print(f"# Path: {path}")
        print(f"{'#'*70}")

        interp = load_interpreter(path)
        weights = get_weight_tensors(interp)

        compare_input_output_quant(interp, label)
        overall_weight_stats(weights, label)
        analyze_depthwise_weights(weights, label)

        results[label] = {'weights': weights, 'interp': interp}

    # Direct comparison if both exist
    if "ST per-tensor (WORKS)" in results and "My per-tensor (BROKEN)" in results:
        print(f"\n{'#'*70}")
        print(f"# DIRECT COMPARISON: ST vs My per-tensor")
        print(f"{'#'*70}")

        st_w = results["ST per-tensor (WORKS)"]['weights']
        my_w = results["My per-tensor (BROKEN)"]['weights']

        print(f"\n  ST model has {len(st_w)} weight tensors")
        print(f"  My model has {len(my_w)} weight tensors")

        # Compare first conv weight scales
        st_first = [w for w in st_w if 'Conv' in w['name'] or 'conv' in w['name']][:1]
        my_first = [w for w in my_w if 'Conv' in w['name'] or 'conv' in w['name']][:1]

        if st_first and my_first:
            print(f"\n  First conv layer comparison:")
            print(f"    ST: shape={st_first[0]['shape']} scale={st_first[0]['scales'][0] if len(st_first[0]['scales']) else 'N/A'}")
            print(f"    My: shape={my_first[0]['shape']} scale={my_first[0]['scales'][0] if len(my_first[0]['scales']) else 'N/A'}")

if __name__ == "__main__":
    main()
