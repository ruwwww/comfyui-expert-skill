# Model Architecture Integration

This covers registering new model architectures in ComfyUI, wrapping foreign models (HuggingFace, diffusers, timm), and how the architecture detection system works.

## How ComfyUI Detects Architecture

When you load a checkpoint, ComfyUI must determine which model architecture it represents. The detection chain is:

1. `model_detection.py` scans all registered architecture classes
2. Each class has a `matches(unet_config, state_dict)` classmethod
3. The first class whose `matches()` returns `True` is selected
4. That class's `get_model()` creates the actual PyTorch model

The registered classes live in `comfy.supported_models.models` — a list in `/comfy/supported_models.py` (near line 2278) containing ~95 model configurations.

## Registering a New Architecture

### When to register a new architecture

- You have a new model that doesn't match any existing ComfyUI architecture
- You want the model to be auto-detected from its checkpoint
- Users should be able to use `Load Checkpoint` or `UNET Loader` with it

### When NOT to register

- You're wrapping a one-off HuggingFace model for experimentation — use manual ModelPatcher wrapping instead
- Your model IS an existing architecture with small modifications — use patching on top of the existing class
- You're creating a node-specific model — just create it in your node's execute method

### Registration pattern

```python
# In your __init__.py or a module imported by it:
from comfy.supported_models import models as SUPPORTED_MODELS
from comfy.supported_models_base import BASE
from comfy import latent_formats
from comfy import model_base

class MyNewArchitecture(BASE):
    unet_config = {
        "model_channels": 320,
        "in_channels": 4,
        "out_channels": 4,
        "num_res_blocks": 2,
        "attention_resolutions": [4, 2, 1],
        "context_dim": 768,
        # ... all config keys used by the UNet constructor
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,  # -1 means: heads = channels // 64
    }

    latent_format = latent_formats.SD15()  # Or your custom format
    sampling_settings = {"sigma_min": 0.0292, "sigma_max": 14.614}
    memory_usage_factor = 2.0

    @classmethod
    def matches(cls, unet_config, state_dict):
        # Check for identifying keys in state_dict
        # Check unet_config values if needed
        if "my_special_layer.weight" in state_dict:
            return True
        return False

    def model_type(self, state_dict, prefix=""):
        return model_base.ModelType.EPS  # Or FLOW, V_PREDICTION, etc

    def clip_target(self, state_dict={}):
        # Return the CLIP model and tokenizer this architecture uses
        from comfy.supported_models_base import ClipTarget
        return ClipTarget(
            tokenizer=comfy.sd1_clip.SDTokenizer,
            clip=comfy.sd1_clip.SDClipModel,
        )

# Register the architecture
SUPPORTED_MODELS.append(MyNewArchitecture)
```

### Required overrides

Every architecture class must override:

- `unet_config` — dict passed to the UNet constructor
- `matches(unet_config, state_dict)` — True if this architecture fits the checkpoint
- `model_type(state_dict, prefix)` — one of `ModelType` enum values
- `clip_target(state_dict)` — which CLIP model and tokenizer to use

Common optional overrides:
- `latent_format` — custom `LatentFormat` subclass for model-specific latent structure
- `get_model(state_dict, prefix, device)` — override if you need a custom `BaseModel` subclass
- `process_clip_state_dict(sd)` / `process_unet_state_dict(sd)` — state dict key remapping
- `process_clip_state_dict_for_saving(sd)` / `process_unet_state_dict_for_saving(sd)` — reverse

### ModelType values

From `/comfy/model_base.py`:

| Type | When to use |
|------|------------|
| `EPS` | Standard epsilon prediction (SD1.5, SD2) |
| `V_PREDICTION` | Velocity prediction (SDXL refiners) |
| `EDM` | EDM-style prediction |
| `FLOW` | Flow matching (SD3, SD4) |
| `FLUX` | Flux-style flow matching |
| `FLOW_COSMOS` | Cosmos flow matching |

The `model_type` determines how the denoising prediction is computed from the model output.

## Custom UNet Construction

If your architecture needs a non-standard UNet, you have two options:

**Option 1: Use an existing UNet via unet_config.** Most architectures just configure one of ComfyUI's built-in UNet classes (SDUNet, Flux, etc) via `unet_config` dict. The UNet class is determined by the config keys — `model_detection.py`'s `model_config_from_unet()` maps config to class.

**Option 2: Custom UNet class.** Override `get_model()`:
```python
def get_model(self, state_dict, prefix="", device=None):
    # Build your custom model
    diffusion_model = MyCustomUNet(**self.unet_config)
    # Load weights
    if state_dict:
        diffusion_model.load_state_dict(state_dict, strict=False)
    # Wrap in BaseModel
    model = model_base.BaseModel(self, self.model_type(state_dict, prefix), device, unet_model=diffusion_model)
    return model
```

## Wrapping HuggingFace / Diffusers Models

For models that come from HuggingFace or another framework:

### Step 1: Load the foreign model

```python
from diffusers import MyDiffusersPipeline

pipe = MyDiffusersPipeline.from_pretrained("org/model-name", torch_dtype=torch.float16)
foreign_model = pipe.transformer  # or pipe.unet
foreign_model.eval()
```

### Step 2: Create a ComfyUI adapter

The foreign model must expose a forward signature compatible with ComfyUI's sampling. Create an adapter if needed:

```python
class ComfyUIAdapter(nn.Module):
    """Wraps a HuggingFace DiT model for ComfyUI's sampling pipeline."""
    def __init__(self, hf_model, latent_format):
        super().__init__()
        self.model = hf_model
        self.latent_format = latent_format
    
    def forward(self, x, timestep, context, y=None, **kwargs):
        # x: [B, C, H, W] latent
        # timestep: [B] timestep values
        # context: conditioning (from transformer_options or model)
        # y: optional guidance embedding
        
        # Call the HF model in its expected format
        noise_pred = self.model(
            hidden_states=x,
            encoder_hidden_states=context,
            timestep=timestep,
            pooled_projections=y,
            return_dict=False,
        )[0]
        
        return noise_pred
```

### Step 3: Wrap in ModelPatcher

```python
import comfy.model_patcher
import comfy.model_management

adapter = ComfyUIAdapter(foreign_model, my_latent_format)

patcher = comfy.model_patcher.ModelPatcher(
    model=adapter,
    load_device=comfy.model_management.get_torch_device(),
    offload_device=comfy.model_management.unet_offload_device(),
)

# Set up model_sampling for sigma/timestep conversion
from comfy.model_sampling import ModelSamplingDiscrete, EPS
patcher.model_sampling = ModelSamplingDiscrete()
# Configure sigma range
patcher.model_sampling.set_sigmas(...)

patcher.model_options = {"transformer_options": {}}
```

### When wrapping, you MUST handle

1. **Forward signature**: ComfyUI calls `model(x, timestep, context, **kwargs)`. If your model expects different kwargs, adapt.
2. **Sigma/timestep conversion**: Your model expects timestep values in a specific range. `model_sampling` handles conversion from ComfyUI's sigma space.
3. **Device management**: The ModelPatcher handles load/unload. Don't manually move the model.
4. **Dtype**: Cast inputs to match model dtype if needed, but let ComfyUI handle model-level dtype.
5. **Conditioning threading**: ComfyUI passes conditioning through `model_options["transformer_options"]` and as `context` argument. Your wrapper must route it correctly.

## Common Pitfalls

**Pitfall 1: Forgetting to register the architecture.** If `matches()` never gets called, your class isn't in the `models` list. Registration must happen at import time (in `__init__.py` or a module it imports).

**Pitfall 2: matcher conflicts.** If your `matches()` is too broad, it may shadow existing architectures. Check the state_dict carefully for unique keys.

**Pitfall 3: Not handling config-less checkpoints.** Some checkpoints don't have `unet_config` in metadata. Your `matches()` should handle `unet_config = {}` gracefully.

**Pitfall 4: Inference dtype.** The architecture config should specify `supported_inference_dtypes`. For FP16 models, include `torch.float16`. For models that only work in BF16, set `manual_cast_dtype = torch.bfloat16`.

**Pitfall 5: Memory usage.** Set `memory_usage_factor` to reflect VRAM needs. Default is 2.0. Video models need 3-4x. Set too low and ComfyUI will OOM; too high and it won't use available VRAM efficiently.