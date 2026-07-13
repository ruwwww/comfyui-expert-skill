# Advanced LoRA Patching & Metadata Reference

This reference document outlines the engineering patterns for applying LoRA layers to quantized weights (Bake-In vs. Dynamic paths), metadata extraction constraints, and `ModelPatcher` cloning requirements.

---

## 🎨 1. LoRA Patching Paradigms for Quantized Models

Because quantized weights (e.g., INT8/INT4 packed formats) cannot accept floating-point addition in-place, developers must select one of two architectural patterns to apply LoRA checkpoints:

```
               [ Input Quantized Weight: W_quant ]
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼ [ Bake-In Path: Load Time ]                   ▼ [ Dynamic Path: Run Time ]
Dequantize to Float                               Keep Weight Quantized
  │                                                │
  ▼                                                ▼
Apply LoRA Delta in Float space                  Forward Pass Matrix Mult:
  │                                                Y = QuantMatMul(X, W_quant) + X @ A @ B
  ▼
Re-quantize back to Target Layout
  │
  ▼
Update Parameter: W_quant_new
```

### A. The Bake-In Path (`dequant-patch-requant`)
This path is executed at **model load time** inside a custom `ModelPatcher`.
1. **Dequantize:** Convert the quantized parameter to its float32 representation:
   $$\mathbf{W}_{\text{float}} = \text{dequantize}(\mathbf{W}_{\text{quant}}, \text{scale})$$
2. **Apply Delta:** Run ComfyUI's native weight calculation to sum the low-rank delta updates:
   $$\mathbf{W}_{\text{patched\_float}} = \text{comfy.lora.calculate\_weight}(\text{patches}, \mathbf{W}_{\text{float}}, \text{key})$$
3. **Re-quantize:** Compress the modified float matrix back to the target format (e.g. `quantize_int8` or `quantize_convrot_w4a4`) and update the module parameter.

* **Pros:** Zero runtime execution overhead; standard forward paths run without changes.
* **Cons:** Introduces load-time latency (dequant/requant calculations) and potential precision loss during the second quantization phase.

### B. The Dynamic Path (Runtime Injection)
This path is executed at **execution time** inside custom forward operations.
1. Cache the LoRA parameters (weights $A$ and $B$, scale factor $\alpha$) directly on the quantized tensor object during loading.
2. In the forward pass, calculate the low-rank product dynamically and add it to the output:
   $$\mathbf{Y} = \text{Linear}_{\text{quantized}}(\mathbf{X}, \mathbf{W}_{\text{quant}}) + \alpha \cdot (\mathbf{X} \mathbf{A}) \mathbf{B}^T$$
* **Pros:** Avoids re-quantization representation loss.
* **Cons:** Increases GEMM execution latency due to additional runtime matrix operations.

---

## 🏷️ 2. Metadata Extraction Without Weight Loading

To check model architectures, configurations, or quantization flags instantly without loading gigabytes of weights into memory:

* **Safetensors Header parsing:** Safetensors files structure metadata as a JSON header at the beginning of the file. Use `safe_open` or ComfyUI's utility to read this header without parsing tensor values:
  ```python
  from safetensors import safe_open
  
  with safe_open("model.safetensors", framework="pt") as f:
      metadata = f.metadata()  # Returns metadata dict instantly (zero tensor memory cost)
  ```
* **ComfyUI Utility:**
  ```python
  sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
  # When return_metadata=True, ComfyUI parses the header safely
  ```

---

## 💾 3. ModelPatcher Cloning Constraints

ComfyUI's native `ModelPatcher.clone()` returns a **fresh patcher object** and does NOT copy arbitrary attributes bound to the source instance.

* **The Problem:** If custom loaders stash metadata (like `_safetensors_metadata` or `_quantization_metadata`) on the patcher instance, these variables are lost when down-stream nodes (like `LoRALoader`) clone the model. When saving checkpoints, the exporter writes corrupted headers.
* **The Solution:** Manually copy custom metadata attributes during loader clone operations:
  ```python
  model_patcher = model.clone()
  for attr in ("_safetensors_metadata", "_quantization_metadata", "_int8_source_metadata"):
      if hasattr(model, attr) and not hasattr(model_patcher, attr):
          setattr(model_patcher, attr, getattr(model, attr))
  ```
