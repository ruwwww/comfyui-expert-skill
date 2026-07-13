# Safetensors Metadata & Model Conversion Reference

This reference document details the JSON schema for ComfyUI's quantization metadata header and the packing/conversion conventions for custom layouts.

---

## 🏷️ 1. ComfyUI Quantization Metadata System

ComfyUI parses the header metadata of `.safetensors` files to detect quantized structures and load the corresponding layouts.

### A. The JSON Header Schema
Metadata must be saved under the key `_quantization_metadata` in the safetensors file header. It contains a map of layer base names to their format settings:

```json
{
  "format_version": "1.0",
  "layers": {
    "net.blocks.1.attn.q_proj": {
      "format": "svd_orbitquant_w4a4"
    },
    "net.blocks.1.mlp.fc1": {
      "format": "svd_orbitquant_w4a4"
    }
  }
}
```

* **`format`:** Must match a key registered in ComfyUI's `QUANT_ALGOS` dictionary (e.g., `"svd_orbitquant_w4a4"`).
* **Layer Keys:** Use the base parameter prefix (typically the layer name excluding the `.weight` suffix).

---

## 📦 2. Parameter Packing Conventions

For custom layouts, weights are packed and saved along with their auxiliary tensors in the model state dict:

| Parameter Key | PyTorch Dtype | Shape / Dimension | Description |
| :--- | :---: | :---: | :--- |
| `*.weight` | `torch.int8` | `(out_features, in_features // 2)` | The packed 4-bit integer weights (2 values per byte). |
| `*.weight_scale` | `torch.bfloat16` | `(out_features, or groups)` | Scaling factors for rows/blocks. |
| `*.weight_proj_down` | `torch.bfloat16` | `(in_features, rank)` | SVD low-rank down-projection bottleneck matrix. |
| `*.weight_proj_up` | `torch.bfloat16` | `(out_features, rank)` | SVD low-rank up-projection matrix. |
| `*.weight_centroids` | `torch.bfloat16` | `(num_centroids,)` | The non-linear codebook lookup table (centroids). |

---

## 💾 3. Writing Safetensors in Python

When exporting quantized weights from a python script, pack the serialization configuration into a flat dictionary and pass it to `safetensors.torch.save_file`:

```python
import json
from safetensors.torch import save_file

# Define the metadata dictionary
metadata = {
    "format": "comfy",
    "_quantization_metadata": json.dumps({
        "format_version": "1.0",
        "layers": {
            "net.blocks.1.mlp.fc1": {"format": "svd_orbitquant_w4a4"}
        }
    })
}

# Save state_dict containing packed int8 tensors and BF16 scales/projections
save_file(quantized_state_dict, "model.safetensors", metadata=metadata)
```
