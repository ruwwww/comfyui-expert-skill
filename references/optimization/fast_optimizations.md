# ComfyUI '--fast' Reference Optimizations

This reference document details the specific performance flags, hardware execution paths, and implementation details of ComfyUI's `--fast` optimizations.

---

## ⚡ 1. Performance Flags Reference

Replicate ComfyUI's native `--fast` performance improvements in standalone scripts using the following runtime configurations:

```python
# 1. Enable mixed-precision accumulation for matmul operations
# (Allows Tensor Cores to accumulate in FP16/BF16 instead of forcing FP32)
torch.backends.cuda.matmul.allow_fp16_accumulation = True
if hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp"):
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# 2. Enable CuDNN benchmark autotuning
# (Tells CuDNN to profile and select the fastest convolution algorithm on startup)
torch.backends.cudnn.benchmark = True

# 3. Apply hardware-accelerated FP8 Matrix Multiplications via torchao
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
quantize_(model, Float8DynamicActivationFloat8WeightConfig())
```

---

## 📈 2. Hardware Execution & Benchmarks

* **FP8 Matrix Multiplication (`fp8_matrix_mult`):** Maps weights and activations onto native Blackwell E4M3/E5M2 Tensor Core formats. In memory-bound layers (like MLP expansion blocks), this cuts memory transfer requirements in half and achieves a **1.58x speedup** on the RTX 5060 Ti GPU.
* **CuBLAS Autotuning (`autotune`):** Speeds up model initialization and repetitive GEMM execution structures by running layout heuristics on start.
