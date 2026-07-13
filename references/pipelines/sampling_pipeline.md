# Sampling Pipeline

This covers ComfyUI's sampling architecture — how noise becomes images, where CFG happens, how sigma schedules work, and where you can inject custom logic.

## Pipeline Overview

The call chain for an image generation is:

```
KSampler node
  → comfy.sample.sample() or sample_custom()
    → comfy.samplers.KSampler
      → KSampler.sample() loop:
        for each sigma in sigmas:
          1. Prepare noise (add noise to latent)
          2. Run CFG:
             a. model_management.load_model_gpu(model)
             b. model.apply_model(noise_for_cond)    → cond_pred
             c. model.apply_model(noise_for_uncond)  → uncond_pred
             d. Run pre_cfg_functions on cond_pred, uncond_pred
             e. cond_pred = uncond_pred + cfg * (cond_pred - uncond_pred)
             f. Run post_cfg_functions on cond_pred
          3. sampler_function(cond_pred, sigma, ...) → get next latent
```

## KSampler

`comfy.samplers.KSampler` (at `/comfy/samplers.py`) is the main sampler class. It is created with:

```python
sampler = comfy.samplers.KSampler(
    model,              # ModelPatcher
    steps=20,
    device=load_device,
    sampler="euler",    # Sampler name string
    scheduler="normal", # Scheduler name string
    denoise=1.0,
    model_options=model.model_options  # Options dict flows through
)
```

The sampler registry lives in `/comfy/samplers.py`'s `KSAMPLER_NAMES` and `SCHEDULER_NAMES` dicts. Custom samplers add themselves to these:

```python
# Register a custom sampler
import comfy.samplers

def my_sampler_function(model, x, timestep, **extra_args):
    # ... custom logic ...
    return model(x, timestep, **extra_args)

comfy.samplers.KSAMPLER_NAMES["my_sampler"] = my_sampler_function
```

## Custom SAMPLER Node Pattern

To create a node that extends the sampler with custom behavior, don't register a sampler name — instead, create a node that uses `sample_custom()` with your own logic wrapping the sampling loop:

```python
import comfy.samplers
import comfy.sample

class MyCustomSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "cfg": ("FLOAT", {"default": 7.0, ...}),
                # ... other params
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
    def sample(self, model, positive, negative, latent_image, cfg, ...):
        model = model.clone()
        
        # Generate noise
        noise = comfy.sample.prepare_noise(latent_image["samples"], seed)
        
        # Set up sigmas
        sigmas = comfy.samplers.calculate_sigmas(model, "normal", steps)
        
        # Define custom sampler that wraps the noise prediction
        def my_custom_sampler(model, x, timestep, **extra_args):
            # Get noise prediction
            out = model(x, timestep, **extra_args)
            # Apply your modification
            out = my_modification(out, timestep)
            return out
        
        # Run sampling with custom sampler
        latent_samples = comfy.sample.sample_custom(
            model, noise, cfg, my_custom_sampler, sigmas,
            positive, negative, latent_image["samples"]
        )
        
        return ({"samples": latent_samples},)
```

## CFGGuider and CFG Hooks

The `CFGGuider` in `/comfy/samplers.py` controls classifier-free guidance. During each step, it:

1. Runs `cond_pred = model(noise_cond, ...)`
2. Runs `uncond_pred = model(noise_uncond, ...)`
3. Applies `sampler_pre_cfg_function` hooks
4. Computes `cfg_result = uncond_pred + cfg_scale * (cond_pred - uncond_pred)`
5. Applies `sampler_post_cfg_function` hooks

### Injecting pre/post-CFG hooks

Use the helper functions from `comfy.model_patcher`:

```python
from comfy.model_patcher import set_model_options_post_cfg_function, set_model_options_pre_cfg_function

def my_post_cfg(args):
    """Called after CFG combination with the CFG result.
    args is a dict with:
        'model': ModelPatcher
        'cond_result': conditional noise prediction
        'uncond_result': unconditional noise prediction  
        'input': the CFG result (uncond + cfg * (cond - uncond))
        'sigma': current noise level
        'model_options': current model options
    Returns the (potentially modified) CFG result.
    """
    output = args["input"]
    # Modify output here...
    return output

model = model.clone()
model.model_options = set_model_options_post_cfg_function(
    model.model_options, my_post_cfg
)
```

This is the pattern used by `ComfyUI-AutomaticCFG` for dynamic CFG scaling and various attention-aware guidance methods.

## Sigma and Noise Schedules

### What sigmas are

Sigmas are the noise levels at each sampling step. They decrease from a high value (maximum noise) to a low value (~0, clean image). The sequence of sigmas is called the "schedule" and is generated by the scheduler.

### Working with sigmas

Sigmas are generated by `comfy.samplers.calculate_sigmas(model, scheduler, steps, denoise)`. Schedulers are registered in `comfy.samplers.SCHEDULER_NAMES`.

Custom sigma manipulation:
```python
def custom_sigmas(model, scheduler, steps, denoise):
    sigmas = comfy.samplers.calculate_sigmas(
        model.model_sampling, scheduler, steps, denoise
    )
    # Modify sigmas here
    sigmas = my_sigma_transform(sigmas)
    return sigmas
```

Common sigma manipulations:
- **Truncation**: `sigmas = sigmas[:n]` to skip early steps
- **Rescaling**: `sigmas = sigmas * scale` to change the noise schedule intensity
- **Non-uniform spacing**: `sigmas = sigmas ** exponent` to bias steps toward early/late stages

### ModelSampling

The `model.model_sampling` object (from `/comfy/model_sampling.py`) knows how to convert between sigmas and timesteps and handles the model's native noise schedule:

```python
# Convert between sigma and timestep
timestep = model.model_sampling.timestep(sigma)
sigma_back = model.model_sampling.sigma(timestep)

# Get model's sigma range
sigmas_max = model.model_sampling.sigma_max
sigmas_min = model.model_sampling.sigma_min
```

## model_wrap and inner_model

During sampling, the model is wrapped to adapt its interface:
- `model_wrap.inner_model.model` is the actual UNet/DiT
- The wrapper handles noise prediction, scaling, and sigma/timestep conversion
- When writing patches that target the model during sampling, `transformer_options` is the right hook point, not direct model access

## Noise Injection

ComfyUI supports masked noise injection for inpainting and regional generation via `noise_mask`. The `LATENT` dict may carry a `"noise_mask"` key. When present:

```python
noise = comfy.sample.prepare_noise(latent_image, seed)
if "noise_mask" in latent_image:
    noise_mask = latent_image["noise_mask"]
    # During sampling, noise is applied only where mask == 1
    latent = latent * (1 - noise_mask) + noise * noise_mask
```

If your custom sampler needs to respect masks, read `"noise_mask"` from the latent dict.

## Testing Sampling Nodes

Always test with a minimal graph first:
1. `EmptyLatentImage` → `YourCustomSampler` or `KSampler` → `VAEDecode` → `PreviewImage`
2. Verify the output is a coherent image (not noise, not black)
3. Then test in a complex workflow with ControlNet, LoRA, etc.

This isolates whether your sampler modification produces valid outputs from whether it interacts correctly with conditioning.