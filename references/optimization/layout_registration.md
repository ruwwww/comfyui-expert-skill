# Custom Layout & Quantization Registration Reference

This reference document details how to register custom quantization layouts and interface them with ComfyUI's loader system.

---

## 🔌 1. Registration Steps (Without Patching Core)

### A. Register Layout Class & Configuration
Register your layout class dynamically within your custom node startup module (e.g., `__init__.py` in `/custom_nodes/`):

```python
import torch
from comfy.quant_ops import QUANT_ALGOS, register_layout_class

# 1. Register the class to the layout registry
register_layout_class("SVDOrbitQuantW4A4Layout", SVDOrbitQuantW4A4Layout)

# 2. Add the algorithm mapping to QUANT_ALGOS
QUANT_ALGOS.setdefault(
    "svd_orbitquant_w4a4",
    {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "weight_proj_down", "weight_proj_up", "weight_centroids"},
        "comfy_tensor_layout": "SVDOrbitQuantW4A4Layout",
        "group_size": 64,
        "quantize_input": False,  # True = quantize activations; False = pass float/bf16 activations directly
    },
)
```

---

## 🛠️ 2. Dynamic Startup Monkeypatching

Because ComfyUI validates format configurations inside `comfy/ops.py` on startup, you must wrap `comfy.ops._load_quantized_module` to parse custom checkpoints manually:

```python
import json
import comfy.ops
original_load = comfy.ops._load_quantized_module

def custom_load(module, super_load, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    layer_conf = state_dict.get(f"{prefix}comfy_quant", None)
    if layer_conf is not None:
        conf_dict = json.loads(layer_conf.numpy().tobytes())
        if conf_dict.get("format") == "svd_orbitquant_w4a4":
            device = module.factory_kwargs["device"]
            compute_dtype = module.factory_kwargs["dtype"]
            
            # Pop weight and custom scales
            weight = state_dict.pop(f"{prefix}weight").to(device=device)
            scale = state_dict.pop(f"{prefix}weight_scale").to(device=device)
            proj_down = state_dict.pop(f"{prefix}weight_proj_down").to(device=device)
            proj_up = state_dict.pop(f"{prefix}weight_proj_up").to(device=device)
            centroids = state_dict.pop(f"{prefix}weight_centroids").to(device=device)
            
            # Bind to layout parameters
            from comfy.quant_ops import QuantizedTensor, get_layout_class
            layout_cls = get_layout_class("SVDOrbitQuantW4A4Layout")
            params = layout_cls.Params(
                scale=scale,
                proj_down=proj_down,
                proj_up=proj_up,
                centroids=centroids,
                orig_dtype=compute_dtype,
                orig_shape=module._orig_shape
            )
            
            module.quant_format = "svd_orbitquant_w4a4"
            module.layout_type = "SVDOrbitQuantW4A4Layout"
            module.weight = torch.nn.Parameter(
                QuantizedTensor(weight, module.layout_type, params),
                requires_grad=False
            )
            
            # Strip manually loaded keys from Comfy's tracking sets
            consumed = [f"{prefix}weight", f"{prefix}weight_scale", f"{prefix}weight_proj_down", f"{prefix}weight_proj_up", f"{prefix}weight_centroids", f"{prefix}comfy_quant"]
            for c in consumed:
                if c in missing_keys:
                    missing_keys.remove(c)
            state_dict.pop(f"{prefix}comfy_quant", None)
            return
            
    return original_load(module, super_load, state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs, load_extra_params)

comfy.ops._load_quantized_module = custom_load
```

---

## ⚡ 3. Operator Registration

To map PyTorch tensor execution onto layout math, register your layout math operations using `@register_layout_op`:

```python
from comfy.quant_ops import register_layout_op

@register_layout_op(torch.ops.aten.linear.default, SVDOrbitQuantW4A4Layout)
def _handle_custom_linear(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    
    # Run your custom quantized GEMM or split math here
    ...
```
