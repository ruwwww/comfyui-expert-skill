# GGUF & GGML Custom Tensor Pattern Reference

This reference document details the advanced architecture for wrapping quantized weights inside custom `torch.Tensor` subclasses, patching low-bit models with dynamic LoRAs, implementing custom `ModelPatcher` behaviors, and managing memory-mapped files inside ComfyUI.

---

## 📦 1. Custom Tensor Subclassing (`GGMLTensor` Pattern)

When loading low-bit quantized weights (like GGUF/GGML formats) that do not match standard PyTorch layouts, wrap them in a custom `torch.Tensor` subclass to preserve architectural shape checks:

```python
import torch

class GGMLTensor(torch.Tensor):
    """
    Custom tensor wrapper to store packed quantized weights while
    exposing the unquantized float dimensions to PyTorch/ComfyUI.
    """
    def __new__(cls, data, *, tensor_type, tensor_shape, patches=None, **kwargs):
        # Creates the underlying raw data storage tensor (typically 1D or packed byte arrays)
        return super().__new__(cls, data, **kwargs)

    def __init__(self, data, *, tensor_type, tensor_shape, patches=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type      # Quantization format indicator (e.g. gguf.GGMLQuantizationType)
        self.tensor_shape = tensor_shape    # Original float shape (e.g., [out_features, in_features])
        self.patches = patches or []        # Cache list to store dynamic LoRA updates

    def to(self, *args, **kwargs):
        # Binds custom attributes across device and dtype casting transformations
        new_tensor = super().to(*args, **kwargs)
        new_tensor.tensor_type = getattr(self, "tensor_type", None)
        new_tensor.tensor_shape = getattr(self, "tensor_shape", new_tensor.data.shape)
        new_tensor.patches = getattr(self, "patches", []).copy()
        return new_tensor

    def clone(self, *args, **kwargs):
        # Override clone to bypass unnecessary deep copying of immutable weights
        return self

    @property
    def shape(self):
        # CRITICAL: Returns the virtual unquantized dimensions instead of the raw packed data size.
        # This prevents shape checks in nn.Linear or Attention layers from crashing.
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape
```

---

## ⚡ 2. Dynamic LoRA Patching on Quantized Tensors

### The Challenge
Standard LoRA updates are calculated as floating-point weight deltas:
$$W_{\text{new}} = W_{\text{orig}} + \alpha \cdot \Delta W_{\text{lora}}$$
Because quantized tensors are packed byte matrices representing indices or centroids, you cannot directly apply floating-point addition in-place on the raw quantized data.

### The Solution (Lazy Runtime Accumulation)
Instead of modifying the weights during model loading, the custom `ModelPatcher` caches the LoRA parameters onto the tensor, and dequantization modules compute and add the delta at execution time:

```python
# 1. Caching Patches (Inside Custom ModelPatcher)
if is_quantized(weight):
    # Bypass standard in-place addition. Assign the LoRA delta layers directly to the tensor patches list.
    weight.patches = [(patches_list, weight_key)]

# 2. Lazy Application (Inside Custom Linear Layer Forward Pass)
def forward(self, x):
    if isinstance(self.weight, GGMLTensor):
        # Step A: Dequantize the packed tensor to float
        w_float = dequantize_tensor_to_float(self.weight, out_dtype=x.dtype)
        
        # Step B: Calculate and add the LoRA delta on the float representation
        if len(self.weight.patches) > 0:
            for patches, key in self.weight.patches:
                w_float = comfy.lora.calculate_weight(patches, w_float, key)
                
        # Step C: Execute GEMM using the patched float weight
        return torch.nn.functional.linear(x, w_float, self.bias)
        
    return torch.nn.functional.linear(x, self.weight, self.bias)
```

---

## 💾 3. ModelPatcher Hooks & Memory-Mapped File Management

To support memory-mapped loading (`mmap`) and prevent virtual memory page locking or resource leakage during model offloading:

1. **Unpatching Cleanup:** Overwrite `unpatch_model` to clear patch references from custom tensors, ensuring garbage collection can free memory during VRAM swapping:
   ```python
   def unpatch_model(self, device_to=None, unpatch_weights=True):
       if unpatch_weights:
           for p in self.model.parameters():
               if isinstance(p, GGMLTensor):
                   p.patches = [] # Clear cached LoRA references
       return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)
   ```
2. **mmap Page Release:** For memory-mapped files, force-cast layers from virtual memory space to physical CUDA/CPU memory during the first load cycle to allow OS page files to release.
