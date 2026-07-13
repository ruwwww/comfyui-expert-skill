# Model Management and Loading

This covers how ComfyUI handles device placement, VRAM management, model loading, and the `ModelPatcher` abstraction. This is the subsystem you'll interact with any time you load a checkpoint, wrap a foreign model, or modify model weights.

## Device Management Architecture

ComfyUI uses a layered device management system. The entry point is `comfy.model_management`, which determines device placement based on `VRAMState`, CLI arguments, and hardware capabilities.

VRAM states (from `/comfy/model_management.py`):
- `DISABLED` — No GPU (CPU-only)
- `NO_VRAM` — Very low VRAM, aggressive offloading
- `LOW_VRAM` — Limited VRAM
- `NORMAL_VRAM` — Standard
- `HIGH_VRAM` — Everything stays on GPU
- `SHARED` — Unified memory (Apple Silicon)

Key device query functions:
- `unet_offload_device()` — where the UNet goes when not in active use (CPU for LOW_VRAM, GPU for HIGH_VRAM)
- `unet_inital_load_device(parameters, dtype)` — where to initially load UNet weights
- `text_encoder_device()` / `text_encoder_offload_device()` — same for text encoders
- `vae_device()` / `vae_offload_device()` — same for VAE
- `intermediate_device()` — where intermediate tensors live during sampling

When writing model loading code, use these functions rather than hardcoding `"cuda"` or `"cpu"`. Users have widely different hardware configurations and your node should respect their settings.

When writing model *patching* code, never move tensors to a specific device yourself — use the device the model is already on, or let ComfyUI's memory manager handle it.

## ModelPatcher

`ModelPatcher` (in `/comfy/model_patcher.py`) is the wrapper around all models in ComfyUI. It provides:

- **Weight management**: tracks which weights are loaded onto which device
- **Patching infrastructure**: `add_patches()`, `set_model_sampler_cfg_function()`, etc
- **Model options threading**: carries `model_options` through the sampling pipeline
- **Cloning**: lightweight clone that shares underlying weights but isolates patches

### Key attributes

```python
mp = ModelPatcher(model, load_device, offload_device)
mp.model              # The underlying torch.nn.Module
mp.model_options      # Dict threaded through sampling pipeline
mp.load_device        # torch.device where model is loaded for execution
mp.offload_device     # torch.device where model goes when idle
mp.object_patches     # Dict of object-level patches (replace whole modules)
mp.patches            # Dict of weight-level patches (LoRA-style additive)
```

### Clone pattern

Always clone the ModelPatcher before modifying it. The original may be shared with other nodes in the graph:

```python
def my_execute_method(self, model, **kwargs):
    model = model.clone()  # Isolate our patches from the rest of the graph
    # ... add patches, modify model_options, etc
    return (model,)
```

The `clone()` method shares the underlying `model` object — only the patching metadata and options dicts are deep-copied. This makes cloning cheap.

### Model options

`model.model_options` is a dict that flows through the entire sampling pipeline. It's the primary hook point for most interventions. Key sub-keys:

```python
model.model_options = {
    "transformer_options": {
        "patches": {},         # Weight patches at transformer block level
        "patches_replace": {}, # Module replacement patches
        "wrappers": {},        # Wrapper callbacks (pather_extension)
        "callbacks": {},       # Lifecycle callbacks (patcher_extension)
        # ... any custom keys your node reads
    },
    "sampler_cfg_function": [],     # Pre-CFG hook functions
    "sampler_post_cfg_function": [], # Post-CFG hook functions
    # ... other options
}
```

Always use the helper functions rather than mutating `model_options` dicts directly:

```python
# Correct:
from comfy.model_patcher import set_model_options_patch_replace, set_model_options_post_cfg_function, create_model_options_clone

model_options = create_model_options_clone(model.model_options)
model_options = set_model_options_patch_replace(model_options, my_attn_patch, "attn1", block_name, block_idx)
model_options = set_model_options_post_cfg_function(model_options, my_cfg_hook)
model.model_options = model_options

# Wrong — these dicts may be shared, causing cross-node interference:
model.model_options["sampler_cfg_function"].append(my_cfg_hook)
```

The helper functions do the copy-on-write for you, ensuring your changes don't leak into other nodes.

## Checkpoint Loading

ComfyUI detects model architecture via `comfy.model_detection` which iterates over all registered classes in `supported_models.py`. Each class has a `matches(unet_config, state_dict)` method that tests whether a given checkpoint belongs to that architecture.

The standard loading flow is:
1. `comfy.sd.load_checkpoint_guess_config()` reads the checkpoint file
2. It extracts `unet_config` from the checkpoint metadata or state dict
3. It calls `model_detection.model_config_from_unet()` which iterates `matches()` on all registered architectures
4. The matched config's `get_model()` creates a `BaseModel` with the architecture's UNet
5. Weights are loaded via `load_model_weights()`
6. A `ModelPatcher` wraps the result

### Wrapping a foreign architecture

When you need to load a model from HuggingFace or another framework and use it in ComfyUI's pipeline, you have two paths:

**Path A: Register a new supported_model class** (for models you want auto-detected from state dict):
```python
# In your package, register before any node runs:
from comfy.supported_models import models as supported_models_list
from comfy.supported_models_base import BASE

class MyForeignModel(BASE):
    unet_config = {...}
    latent_format = ...
    def matches(unet_config, state_dict):
        return "my_custom_key" in state_dict

supported_models_list.append(MyForeignModel)
```

**Path B: Manual ModelPatcher wrapping** (for one-off or experimental models):
```python
import comfy.model_patcher
import comfy.model_management

my_hf_model = load_from_huggingface(...)  # Your loading code
my_hf_model.eval()

# Wrap it in a ModelPatcher so it plays nicely with ComfyUI
patcher = comfy.model_patcher.ModelPatcher(
    model=my_hf_model,
    load_device=comfy.model_management.get_torch_device(),
    offload_device=comfy.model_management.unet_offload_device(),
)

# Set up model_options for sampling compatibility
patcher.model_options = {
    "transformer_options": {},
}
```

The model must have:
- A `diffusion_model` attribute OR implement the expected forward signature: `forward(x, timestep, context, **kwargs)`
- `model_sampling` for sigma schedule handling
- `latent_format` for VAE compatibility

If your foreign model doesn't match these, you may need an adapter class between it and the ComfyUI sampling pipeline.

## Quantized Models (GGUF)

GGUF-quantized models store tensors in quantized format. The `ComfyUI-GGUF` pattern (from `/comfy/quant_ops.py` and the GGUF reference repo) shows:

1. Tensors are read as `GGMLTensor` (a subclass of `torch.Tensor`) that stores quantized data
2. Dequantization happens on-the-fly during forward pass via custom ops
3. Model is wrapped in a special `ModelPatcher` that understands quantized weights

```python
# Pattern: register a custom weight computation
import comfy.model_patcher

def patcher(self, n, context_attn1=0, context_attn2=0, **kwargs):
    # Called for each transformer block during sampling
    # 'n' is the block number
    return context_attn1, context_attn2

patcher_model = model.clone()
patcher_model.set_model_attn1_patch(patcher)  # Self-attention
patcher_model.set_model_attn2_patch(patcher)  # Cross-attention
```

## Memory Management

ComfyUI's memory model:
- **Soft offloading**: Models move to `offload_device` when idle, `load_device` when needed. This happens transparently via `model_management.load_models_gpu()`.
- **Low VRAM mode**: Only the currently executing module's weights are on GPU. Everything else is swapped to CPU.
- **Weight sharing**: `ModelPatcher.clone()` shares weights; clones consume negligible extra VRAM.

When writing a node that touches model internals:
- Don't force models onto a specific device — let `model_management` handle it
- Don't hold references to model weights between executions (prevents offloading)
- Use `model_management.load_model_gpu(model_patcher)` to ensure the model is ready before accessing weights