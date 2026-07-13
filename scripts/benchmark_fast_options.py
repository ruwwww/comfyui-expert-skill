import os
import sys

# Crucial: Disable MSLK C++ bindings search which fails on uncompiled local directory
os.environ["MSLK_PYTHON_ONLY"] = "1"

import torch
import torch.nn as nn
import time
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def benchmark_linear(shape, dtype=torch.bfloat16, warmup=10, iterations=50, fp8=False):
    M, K, N = shape
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    linear = nn.Linear(K, N, bias=False, device="cuda", dtype=dtype)
    
    if fp8:
        # Dynamically quantize weight to FP8
        quantize_(linear, Float8DynamicActivationFloat8WeightConfig())
        
    # Warmup
    for _ in range(warmup):
        _ = linear(x)
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iterations):
        _ = linear(x)
    torch.cuda.synchronize()
    
    elapsed = (time.time() - start_time) * 1000 / iterations
    return elapsed

def main():
    print_flush("=" * 60)
    print_flush("🚀 ComfyUI '--fast' Reference Optimizations Benchmark")
    print_flush("=" * 60)
    print_flush(f"GPU: {torch.cuda.get_device_name(0)}")
    
    shapes = [
        (1024, 2048, 2048),  # Attention
        (1024, 2048, 8192),  # MLP
    ]
    
    # Baseline
    print_flush("\n[1] Running Baseline (No fast optimizations)...")
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp"):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        
    baseline_results = []
    for shape in shapes:
        t = benchmark_linear(shape)
        baseline_results.append(t)
        print_flush(f"  Shape {shape}: {t:.4f} ms")
        
    # Enable Fast Accumulation Flags
    print_flush("\n[2] Enabling '--fast' Accumulation Flags...")
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp"):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        
    fast_flags_results = []
    for shape in shapes:
        t = benchmark_linear(shape)
        fast_flags_results.append(t)
        print_flush(f"  Shape {shape}: {t:.4f} ms")
        
    # Enable FP8 Matrix Multiplication
    print_flush("\n[3] Enabling '--fast' FP8 Matrix Multiplication...")
    fp8_results = []
    for shape in shapes:
        t = benchmark_linear(shape, fp8=True)
        fp8_results.append(t)
        print_flush(f"  Shape {shape}: {t:.4f} ms")
        
    # Print comparison
    print_flush("\n" + "=" * 60)
    print_flush("📈 Optimization Summary")
    print_flush("=" * 60)
    for i, shape in enumerate(shapes):
        b = baseline_results[i]
        fl = fast_flags_results[i]
        f8 = fp8_results[i]
        
        speedup_flags = b / fl
        speedup_fp8 = b / f8
        
        print_flush(f"Shape {shape}:")
        print_flush(f"  Baseline:            {b:.4f} ms")
        print_flush(f"  Accumulation Flags:  {fl:.4f} ms (Speedup: {speedup_flags:.2f}x)")
        print_flush(f"  FP8 Matrix Mult:     {f8:.4f} ms (Speedup: {speedup_fp8:.2f}x)")

if __name__ == "__main__":
    main()
