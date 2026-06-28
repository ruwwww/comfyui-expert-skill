# Latent Space Operations

This covers the `LATENT` type contract, VAE encode/decode, latent manipulation, and mask-aware operations. Latent bugs are subtle — a shape mismatch or missing key produces garbage outputs with no obvious error.

## The LATENT Type Contract

`LATENT` is always a dict, never a raw tensor:

```python
latent = {
    "samples": tensor,  # Required. The latent representation.
    # Optional keys:
    "noise_mask": tensor,   # Mask for noise injection (inpainting)
    "batch_index": tensor,  # Batch ordering index
}
```

**Always preserve unknown keys.** If you receive a LATENT with keys beyond `"samples"`, pass them through to your output unchanged:

```python
def process(self, latent, **kwargs):
    samples = latent["samples"]
    # Your processing...
    result = {"samples": processed_samples}
    # Pass through other keys
    result.update({k: v for k, v in latent.items() if k != "samples"})
    return (result,)
```

The `"samples"` tensor has shape `[batch, channels, height, width]` (BCHW). The number of channels depends on the model's latent format — typically 4 for SD/SDXL, 16 for Flux, variable for video models. Channel count is model-specific and should never be hardcoded.

## Latent Format

`latent_formats.py` defines how latents are structured for each model. Read the relevant format class when working with model-specific latents:

```python
from comfy.latent_formats import SDXL, Flux

# Key attributes:
latent_format.latent_channels      # e.g., 4 for SD, 16 for Flux
latent_format.spacial_downscale_ratio  # e.g., 8 for SD (512 → 64 latent)
latent_format.temporal_downscale_ratio  # For video models
latent_format.shift_factor           # For Flux-style shifting
```

`comfy.sample.fix_empty_latent_channels()` adjusts latent dimensions to match the expected format — use it when your node creates fresh latents from scratch.

## VAE Encode/Decode

### Standard pattern

```python
# Encode pixel space → latent
def vae_encode(vae, pixels):
    # pixels: BHWC float32 [0,1] IMAGE format
    latent = vae.encode(pixels[:,:,:,:3])  # vae expects BCHW
    return {"samples": latent}

# Decode latent → pixel space
def vae_decode(vae, latent):
    samples = latent["samples"]
    pixels = vae.decode(samples)
    # pixels: BCHW float32
    pixels = pixels.permute(0, 2, 3, 1)  # Convert to BHWC for IMAGE type
    return (pixels,)
```

### TAESD (Tiny AutoEncoder)

For preview-quality fast decoding, use `taesd`:
```python
# TAESD is selected automatically if available and the VAE is a TAESD model
# No special handling needed — just use the VAE object you receive as input
```

### When to use which

- Full VAE: final output, high quality
- TAESD: previews, intermediate steps, faster iteration
- Your node should accept a VAE input and not dictate which one the user connects

## Batch Operations

Latent tensors are batched along dimension 0. Common batch operations:

```python
# Concatenate two latents (same channels, any spatial size)
samples_a = latent_a["samples"]  # [B, C, H, W]
samples_b = latent_b["samples"]  # [B', C, H, W]
combined = torch.cat([samples_a, samples_b], dim=0)  # [B+B', C, H, W]

# Split batches
for i in range(samples.shape[0]):
    single = {"samples": samples[i:i+1]}

# Repeat to match batch size
target_batch = some_conditioning_tensor.shape[0]
samples = samples.repeat(target_batch, 1, 1, 1)
```

## Spatial Operations

Upscaling/downscaling latents:

```python
import comfy.utils

# Upscale latent (before VAE decode — gives better results than pixel-space upscale)
samples = comfy.utils.common_upscale(
    samples, target_width, target_height,
    upscale_method="bilinear", crop="disabled"
)

# Downscale latent
samples = comfy.utils.common_upscale(
    samples, target_width, target_height,
    upscale_method="area", crop="disabled"
)
```

For tiled operations (large images that exceed VRAM), see the TiledDiffusion and TiledVAE patterns — they process the latent in overlapping tiles and blend the results.

## Mask-Aware Latent Processing

The `"noise_mask"` key enables inpainting-style operations. When present:

```python
def process_with_mask(self, latent, mask, **kwargs):
    samples = latent["samples"]
    noise_mask = latent.get("noise_mask", None)
    
    # If you're creating a new noise_mask:
    # Shape: [batch, 1, height, width] (matches latent spatial dims)
    # Values: 1.0 = keep original, 0.0 = replace with noise
    
    if noise_mask is not None:
        # Apply mask-aware processing
        # Combine your operation's mask with the existing noise_mask
        combined_mask = noise_mask * your_mask
        result = {"samples": processed, "noise_mask": combined_mask}
    else:
        result = {"samples": processed}
    
    return (result,)
```

Mask dimensions: The `"noise_mask"` tensor has shape `[B, 1, H, W]` matching `samples` spatial dims. For video latents (`[B, C, T, H, W]`), masks may need temporal dimension handling — check the parent latent format.

## Nested Tensors

ComfyUI supports "nested tensors" for batching heterogeneous shapes (different spatial dimensions in one batch). Check for nested tensors before doing shape-dependent operations:

```python
from comfy.nested_tensor import NestedTensor

if isinstance(samples, NestedTensor):
    # Handle each sub-tensor separately
    results = []
    for subtensor in samples.unbind():
        result = process_single(subtensor)
        results.append(result)
    samples = NestedTensor(results)
else:
    samples = process_single(samples)
```

## Common Pitfalls

**Pitfall 1: Returning a tensor instead of a dict.** This is the #1 latent bug. Every node that outputs `("LATENT",)` must return `({"samples": tensor},)` not `(tensor,)`. The downstream node expects dict lookup; it will crash or silently produce garbage.

**Pitfall 2: Hardcoding channel count.** `samples.shape[1]` is model-dependent. Never assume it's 4. Use `latent_format.latent_channels` or just process whatever shape you receive.

**Pitfall 3: Forgetting float32 range.** Latent tensors are unnormalized (not [0,1] like images). Don't clamp them to [0,1] — that destroys the latent representation.

**Pitfall 4: Discarding the `noise_mask`.** If your node receives a latent with a noise_mask and doesn't pass it through, inpainting workflows break. Always propagate all extra keys.

**Pitfall 5: Spatial dimension mismatch with masks.** If you resize the `samples` tensor, you must also resize any `noise_mask` to match. Use `comfy.utils.common_upscale()` for the mask too.