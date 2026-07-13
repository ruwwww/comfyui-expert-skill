# comfy-kitchen Integration & Hardware Mapping Reference

This reference document details the connection points between the `comfy-kitchen` performance backend and ComfyUI's loader system, along with hardware-to-backend dispatch mappings across GPU architectures.

---

## 🔌 1. The comfy-kitchen to ComfyUI Touch Points

```
               [ ComfyUI Model Loader (comfy/ops.py) ]
                                 │
                                 ▼ Reads Safetensors Metadata
               [ Layout Registry (comfy/quant_ops.py) ]
                                 │
                                 ▼ Binds QuantizedTensor
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
[ Custom Layout Dispatch ]                       [ comfy-kitchen C++ Backend ]
  E.g. SVDOrbitQuantW4A4Layout                     E.g. convrot_w4a4_linear()
  Calls custom tensor splittings.                 Invokes optimized CUDA GEMMs.
```

1. **Quantization Registry (`QUANT_ALGOS`):** 
   ComfyUI registers layouts inside `comfy/quant_ops.py`. `comfy-kitchen` provides the underlying layout implementations (defining parameters, shapes, and strides).
2. **Tensor Abstraction (`QuantizedTensor`):** 
   During checkpoint loading, ComfyUI wraps weight tensors in a `QuantizedTensor` bound to a specific layout class.
3. **Execution Dispatch (`register_layout_op`):** 
   When PyTorch runs operations (like `torch.ops.aten.linear.default`) on a `QuantizedTensor`, it redirects execution to the layout's registered operator handler, which dispatches down to `comfy-kitchen`'s compiled CUDA operators.

---

## ⚙️ 2. Hardware Architecture & Device Backend Mappings

`comfy-kitchen` adjusts its execution paths depending on the hardware platform, installed GPU driver architecture, and Triton compiler availability:

| Hardware Platform | Compute Backend | Native Backend Path | Execution Properties |
| :--- | :---: | :--- | :--- |
| **NVIDIA Blackwell** | `sm_100` / `sm_101` | **NVFP4 / MXFP8** | Native hardware support for 4-bit and 8-bit block-scaled floating point formats. Near-zero dequantization latency. |
| **NVIDIA Hopper / Ada** | `sm_90` / `sm_89` | **FP8 (E4M3 / E5M2)** | Utilizes standard dynamic FP8 scaling. GEMMs execute directly on FP8 Tensor Cores. |
| **NVIDIA Ampere** | `sm_80` | **Triton / C++ Kernels** | Fallback to optimized JIT Triton kernels or C++ extensions for ConvRot / FHT layouts. |
| **AMD GPUs (ROCm)** | ROCm / HIP | **Triton Kernels (HIP)** | **Triton backend serves as the high-performance cross-platform path.** Compiles Python-based Triton kernels dynamically into HIP code at runtime. |
| **Apple Silicon (MPS)** | MPS (Metal) | **Metal Eager Emulation** | Native Triton/CUDA custom kernels are disabled. Layout operations execute via Apple Metal-supported eager PyTorch operators. |
| **Intel Arc/Data Center** | XPU / OneAPI | **Triton / SYCL Emulation** | Runs via standard PyTorch CPU/XPU fallback operators, or compiles via Triton Intel backend if configured. |
| **CPU** | Host Processor | **CPU Eager Fallback** | Executes layout calculations entirely in eager dequantization pathways on host threads. |
