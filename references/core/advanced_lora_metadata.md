# Advanced LoRA Patching & Metadata Core Reference

This core reference document outlines the engineering patterns for customizing ComfyUI's model patching lifecycle, handling LoRA modifications on non-standard weight formats, stashing metadata across clone boundaries, and parsing headers without weight loading.

---

## 🔌 1. ModelPatcher Inheritance & Customization

The class `comfy.model_patcher.ModelPatcher` manages weight device allocations (VRAM offloading) and dynamic modifications. For custom quantizations or architectural extensions, subclass it directly to intercept loading logic:

```python
import comfy.model_patcher

class CustomModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        # Intercept device transfer to apply custom dequant or patch allocation
        ...
        return super().patch_weight_to_device(key, device_to, inplace_update)

    def clone(self, *args, **kwargs):
        # Ensure cloning preserves CustomModelPatcher type binding
        src_cls = self.__class__
        self.__class__ = CustomModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = CustomModelPatcher
        self.__class__ = src_cls
        return n
```

---

## ⚡ 2. LoRA Patching Paradigms on Quantized Tensors

Because low-bit quantized weights are packed byte formats, standard floating-point addition cannot be performed in-place. Developers must use one of two integration pathways:

### A. The Bake-In Path (`dequant-patch-requant`)
Executed during model load time.
1. **Dequantize:** Convert the quantized parameter to float32:
   $$\mathbf{W}_{\text{float}} = \text{dequantize}(\mathbf{W}_{\text{quant}}, \text{scale})$$
2. **Apply Delta:** Run standard weight calculations to apply LoRA/DoRA modifications on the float representation:
   $$\mathbf{W}_{\text{patched\_float}} = \text{comfy.lora.calculate\_weight}(\text{patches}, \mathbf{W}_{\text{float}}, \text{key})$$
3. **Re-quantize:** Re-compress the updated float matrix back to the target quantized layout and update the module parameter.

* **Pros:** Zero runtime execution overhead; standard execution pathways require no modifications.
* **Cons:** Load-time latency due to dequant/requant iterations, and representation loss during re-quantization.

### B. The Dynamic Path (Runtime Injection)
Executed during the forward pass.
1. Cache the LoRA parameters (weights $A$ and $B$, scale factor $\alpha$) in the tensor's `patches` attribute at load time.
2. In the custom layer's `forward()`, dequantize the weight on-the-fly and execute:
   $$\mathbf{Y} = \text{Linear}_{\text{quantized}}(\mathbf{X}, \mathbf{W}_{\text{quant}}) + \alpha \cdot (\mathbf{X} \mathbf{A}) \mathbf{B}^T$$

* **Pros:** Preserves maximum numerical fidelity (no second quantization step).
* **Cons:** Increases forward pass execution latency due to extra runtime GEMM steps.

---

## 🪝 3. Global `calculate_weight` Interception

To adjust dynamic calculations globally (e.g. modifying channels or scaling dynamically), hook `ModelPatcher.calculate_weight` at startup:

```python
import comfy.model_patcher

# Stash the original implementation
original_calc = comfy.model_patcher.ModelPatcher.calculate_weight

def patched_calc(patches, weight, key):
    # Perform custom channel adjustments or scale mappings here
    ...
    return original_calc(patches, weight, key)

# Override globally
comfy.model_patcher.ModelPatcher.calculate_weight = patched_calc
```

---

## 💾 4. Metadata Stashing & Clone Propagation

ComfyUI's standard `ModelPatcher.clone()` creates a new instance but **does not copy custom attributes** stashed on the parent patcher.

* **The Problem:** If custom loader nodes store file headers or architecture config schemas (e.g. `_safetensors_metadata` or `_int8_source_metadata`) on the patcher, these fields are lost during cloning (which happens whenever LoRAs or patches are applied). Consequently, down-stream save checkpoints write corrupted or incomplete files.
* **The Solution:** Manually propagate metadata attributes across clone boundaries:
  ```python
  model_patcher = model.clone()
  for attr in ("_safetensors_metadata", "_quantization_metadata", "_int8_source_metadata"):
      if hasattr(model, attr) and not hasattr(model_patcher, attr):
          setattr(model_patcher, attr, getattr(model, attr))
  ```

---

## 🏷️ 5. Zero-Load Header Metadata Parsing

To read quantization layouts or model-specific configuration metrics without reading gigabytes of parameters into memory, parse the file's JSON header directly:

```python
from safetensors import safe_open

with safe_open("model.safetensors", framework="pt") as f:
    # Returns metadata dictionary instantly without parsing any tensor data arrays
    metadata = f.metadata() 
```
Alternatively, leverage ComfyUI's loader helper:
```python
sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
# Extracts the header dictionary into the metadata object safely
```
