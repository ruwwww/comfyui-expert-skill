# Node Registration and Type System

This covers how ComfyUI discovers, validates, and executes custom nodes. Mistakes here cause silent failures — a node that doesn't appear in the UI or returns the wrong type will waste hours of debugging.

## Discovery

ComfyUI loads custom nodes by importing each directory under `custom_nodes/`. The `__init__.py` must export `NODE_CLASS_MAPPINGS` and optionally `NODE_DISPLAY_NAME_MAPPINGS`:

```python
# __init__.py
from .nodes import MyNode, MyOtherNode

NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
    "MyOtherNode": MyOtherNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode": "My Node (PackName)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

The key in `NODE_CLASS_MAPPINGS` is the internal node name — it must be unique across all installed custom node packs. Name collisions silently override previous nodes. Always prefix or namespace your node names to avoid this: `"MyPack_MyNode"`, not `"MyNode"`.

## Required class attributes

Every node class defines a `classmethod INPUT_TYPES` and class attributes:

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("output_image", "mask")
    FUNCTION = "process"
    CATEGORY = "image/MyPack"
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = (False, False)
```

### INPUT_TYPES

Must return a dict with `"required"` key and optionally `"optional"` and `"hidden"`.

Input type choices and their spec format:

| Type string | Spec format | Notes |
|---|---|---|
| `"MODEL"` | `("MODEL",)` | ModelPatcher object |
| `"CLIP"` | `("CLIP",)` | CLIP model object |
| `"VAE"` | `("VAE",)` | VAE object |
| `"CONDITIONING"` | `("CONDITIONING",)` | List of [tensor, dict] pairs |
| `"LATENT"` | `("LATENT",)` | Dict with "samples" key |
| `"IMAGE"` | `("IMAGE",)` | BHWC float32 tensor |
| `"MASK"` | `("MASK",)` | B1HW float32 tensor |
| `"INT"` | `("INT", {"default": 0, "min": 0, "max": 100})` | Integer widget |
| `"FLOAT"` | `("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})` | Float widget |
| `"STRING"` | `("STRING", {"default": "", "multiline": True/False})` | String widget |
| `"BOOLEAN"` | `("BOOLEAN", {"default": False})` | Boolean toggle |

Dynamic/discovered inputs (like file lists) are computed by returning a function instead of a dict — see the `"INT"` combo pattern below.

### RETURN_TYPES

A tuple of type strings matching the input type vocabulary above. Must match the number of values returned by your `FUNCTION`. Missing or mismatched return types cause silent node failures.

### RETURN_NAMES

Display names for output slots. Optional but strongly recommended — unnamed outputs are confusing in complex workflows.

### OUTPUT_IS_LIST

Tuple of booleans, one per return type. When `True`, the output is treated as a list — useful for nodes that produce variable-length outputs (batch split, etc).

### OUTPUT_NODE

When `True`, this node is treated as a terminal output (save image, preview, etc). Its outputs are not connected to downstream nodes.

### CATEGORY

Dot-separated path like `"image/MyPack"` or `"latent"`. First segment chooses the top-level menu. Convention is `"<domain>/<pack_name>"` for pack-specific nodes, or bare `"<domain>"` if your node fits naturally into an existing category.

### FUNCTION

Name of the method that executes the node. By convention this is called the "execute method."

### IS_CHANGED

Optional method that returns a hashable value (string, int, tuple). ComfyUI skips re-execution when `IS_CHANGED` returns the same value as last time. Add this when:

- Your node has randomness that should trigger re-execution
- Your node reads external files that might change
- Your node has hidden state not captured by input widgets

```python
def IS_CHANGED(self, seed, **kwargs):
    return float("NaN")  # Always re-execute (random)
    # or
    return os.path.getmtime(some_file)  # Re-execute when file changes
```

## The execute method

The method named by `FUNCTION` receives keyword arguments matching the `INPUT_TYPES` keys. Names in `required` and `optional` dicts become parameter names:

```python
def process(self, image, strength=1.0, model=None, mask=None):
    # image: BHWC float32 tensor [0,1]
    # strength: float from widget
    # model: ModelPatcher
    # mask: BHW float32 tensor or None
    ...
    return (output_image, output_mask)
```

The return value must be a tuple with exactly `len(RETURN_TYPES)` elements. Each element must match the declared type.

## Common type contracts

### IMAGE
- Shape: `[batch, height, width, channels]` — BHWC, float32, range [0, 1]
- NOT CHW, NOT uint8, NOT [0, 255]

### MASK
- Shape: `[batch, 1, height, width]` or `[batch, height, width]` — float32
- Range [0, 1], where 1 = masked/preserved

### LATENT
- Dict with `"samples"` key (a tensor)
- May have additional keys: `"noise_mask"`, `"batch_index"`
- Always pass through keys you don't modify

### CONDITIONING
- List of `[tensor, metadata_dict]` pairs
- metadata_dict may contain `"pooled_output"`, `"control"`, `"gligen"`, etc
- Concatenation: combine lists from multiple conditioning inputs
- Zero-ing out: return `[[torch.zeros_like(cond[0]), cond[1]] for cond in conditioning]`

### MODEL
- A `ModelPatcher` instance from `comfy.model_patcher`
- Has `load_device`, `offload_device`, `model`, `model_options`
- Always `.clone()` before modifying

### CLIP
- A CLIP model object with `encode_from_tokens` and `load_clip` methods
- Usually passed through as-is unless you're doing text embedding manipulation