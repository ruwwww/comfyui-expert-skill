# Dataflow and Activation Patching

This covers how to intercept, modify, or replace model activations during the forward pass — the technique behind TeaCache, DeepCache, token merging, and any paper that modifies intermediate features during inference.

## The Hook System

ComfyUI has a layered hook system for modifying model behavior during inference. The layers, from highest to lowest level:

1. **Wrapper functions** (`patcher_extension.WrappersMP`) — wrap entire call chains
2. **Callbacks** (`patcher_extension.CallbacksMP`) — events at lifecycle points
3. **Transformer options hooks** (`transformer_options`) — per-block callbacks
4. **Module patching** (`set_model_patch` / `set_model_patch_replace`) — replace specific modules
5. **Raw PyTorch hooks** — `register_forward_hook` on specific modules

Choose the highest level that gives you enough control. Lower-level hooks are more fragile and harder to maintain across ComfyUI versions.

## Wrappers (patcher_extension.WrappersMP)

Wrappers are function decorators that wrap entire operations. They let you run code before/after a major pipeline step.

Available wrapper points (from `/comfy/patcher_extension.py`):

| Wrapper | Wraps | Used for |
|---------|-------|----------|
| `OUTER_SAMPLE` | The outermost sampling call | Global sampling hooks, logging |
| `PREPARE_SAMPLING` | Sampling preparation | Modify noise/latent before sampling starts |
| `SAMPLER_SAMPLE` | Each sampling step | Per-step logging or modification |
| `PREDICT_NOISE` | Model noise prediction | Pre/post noise prediction hooks |
| `CALC_COND_BATCH` | Conditional batch calculation | Modify conditioning before application |
| `APPLY_MODEL` | Model forward pass | Pre/post model forward hooks |
| `DIFFUSION_MODEL` | Diffusion model inner forward | Wrap the DiT/UNet forward itself |

### Usage pattern

```python
from comfy.patcher_extension import WrappersMP, add_wrapper_with_key

def my_diffusion_model_wrapper(original_forward, model, params_dict, *args, **kwargs):
    """Wraps the diffusion model's forward pass.
    
    original_forward: callable — the actual forward()
    model: ModelPatcher
    params_dict: parameters passed to forward
    """
    # Pre-forward: modify params
    x = params_dict.get("input")
    timestep = params_dict.get("timestep")
    
    # Call original forward
    result = original_forward(model, params_dict, *args, **kwargs)
    
    # Post-forward: modify result
    # result is the noise prediction tensor
    
    return result

# Register the wrapper
model = model.clone()
add_wrapper_with_key(
    model.model_options,
    WrappersMP.DIFFUSION_MODEL,
    my_diffusion_model_wrapper,
    "my_unique_key"
)
```

### The original_forward signature

Each wrapper type has a different `original_forward` signature. Read `/comfy/patcher_extension.py` and grep for the wrapper's actual usage to understand the exact call signature. Wrappers are calling conventions, not interfaces — they depend on the code that invokes them.

## Callbacks (patcher_extension.CallbacksMP)

Callbacks fire at lifecycle events — model loading, cloning, detaching, etc.

| Callback | Trigger | Used for |
|----------|---------|----------|
| `ON_CLONE` | ModelPatcher.clone() | Re-setup patches on cloned model |
| `ON_LOAD` | Model loaded onto GPU | Initialize GPU-side state |
| `ON_DETACH` | Model offloaded to CPU | Clean up GPU state |
| `ON_CLEANUP` | Model being removed | Release resources |
| `ON_PREPARE_STATE` | Before sampling starts | Setup sampling-time state |
| `ON_APPLY_HOOKS` | Hooks being applied | Modify hook behavior |
| `ON_INJECT_MODEL` | Model injected | Setup injection state |
| `ON_EJECT_MODEL` | Model ejected | Teardown injection state |

### Usage pattern

```python
from comfy.patcher_extension import CallbacksMP, add_callback_with_key

def on_load_callback(patcher):
    """Called when the model is loaded onto GPU."""
    # Rebuild any GPU-side caches or state
    patcher.my_cache = {}

add_callback_with_key(
    model.model_options,
    CallbacksMP.ON_LOAD,
    on_load_callback,
    "my_unique_key"
)
```

## Transformer Options Patching

For block-level granularity within the UNet/DiT, use `transformer_options`:

### patches_replace

Replace attention or MLP computation at specific blocks. This is the attention patching mechanism — detailed in `references/attention_patching.md`. The same mechanism can target non-attention operations if the model's forward pass checks for `patches_replace` at those points:

```python
# General pattern for replacing a module's computation:
from comfy.model_patcher import set_model_options_patch_replace

class MyMLPReplace:
    def __call__(self, *args, extra_options, **kwargs):
        # Read the block info
        block = extra_options["block"]
        # Do custom computation
        ...
        return result
    
    def to(self, device, dtype=None):
        return self

model_options = set_model_options_patch_replace(
    model.model_options,
    MyMLPReplace(),
    "ff",      # Name of the operation to replace
    "input",   # Block group
    n          # Block number
)
```

### Custom keys in transformer_options

You can store arbitrary data in `transformer_options` and read it in your wrappers or patches. This is how to thread configuration from your node into the sampling pipeline:

```python
model.model_options["transformer_options"]["my_node_data"] = {
    "cache": {},
    "threshold": 0.5,
    "enabled": True,
}
```

In your patch or wrapper:
```python
transformer_options = extra_options if "block" in extra_options else kwargs.get("transformer_options", {})
my_data = transformer_options.get("my_node_data", {})
if my_data.get("enabled"):
    ...
```

## Direct Module Patching (set_model_patch)

For replacing entire modules rather than just their computation:

```python
# Replace the input_blocks completely
model.set_model_patch(my_custom_input_blocks, "diffusion_model.input_blocks")
# Replace a specific layer
model.set_model_patch(my_custom_layer, "diffusion_model.middle_block.1")
```

Module paths use dot notation from the model root. Find paths by printing module names:

```python
for name, module in model.model.diffusion_model.named_modules():
    if "attention" in name.lower():
        print(name)
```

## Raw PyTorch Hooks

Use `register_forward_hook` as a last resort when wrapper/transformer_options don't give you the granularity you need:

```python
def my_forward_hook(module, input, output):
    # module: the specific module
    # input: tuple of input tensors  
    # output: the module's output tensor(s)
    return my_transform(output, input)

# Find the target module
target = model.model.diffusion_model.input_blocks[0][0]
handle = target.register_forward_hook(my_forward_hook)

# Remove when done
handle.remove()
```

Hooks are dangerous because:
- They persist across all forward passes — must be manually removed
- They survive model offload/load cycles
- They're not scoped to a single sampling call
- Multiple hooks can accumulate and interfere

Only use raw hooks for temporary debugging or when the module patching system doesn't support your use case.

## Caching Activations (TeaCache / DeepCache Pattern)

The pattern for activation caching is:

1. Wrap the diffusion model's forward with a `WrappersMP.DIFFUSION_MODEL` wrapper
2. In the wrapper, maintain a cache (dict) keyed by timestep
3. On each forward call, compute a similarity metric between current and cached activations
4. If similar enough, skip computing the full model forward — reuse cached output

```python
from comfy.patcher_extension import WrappersMP, add_wrapper_with_key

def teacache_wrapper(original_forward, model, params_dict, *args, **kwargs):
    transformer_options = params_dict.get("transformer_options", {})
    if not transformer_options.get("enable_teacache", False):
        return original_forward(model, params_dict, *args, **kwargs)
    
    timestep = params_dict.get("timestep")
    cache = transformer_options.setdefault("_teacache", {})
    
    # Check if we should skip
    if should_skip(timestep, cache):
        return cache.get("last_output")
    
    result = original_forward(model, params_dict, *args, **kwargs)
    
    # Update cache
    cache["last_output"] = result
    cache["last_timestep"] = timestep
    
    return result
```

The TeaCache reference repo (`ComfyUI-TeaCache` in the workspace) demonstrates this pattern across multiple model architectures (Flux, LTX Video, Wan).

## Gating Mechanisms (Paper Implementation Pattern)

When a paper adds a gating mechanism to FFN or attention outputs:

1. Wrap `DIFFUSION_MODEL` with your gate logic
2. In the wrapper, compute the gate value based on current activations
3. Apply the gate to the original model output

This avoids needing to modify the model architecture itself — you operate on the output.

## Debugging Activation Shapes

When implementing a patch that depends on tensor shapes, add temporary debug hooks to verify your assumptions:

```python
def debug_hook(module, input, output):
    if isinstance(output, torch.Tensor):
        print(f"  {module.__class__.__name__}: output.shape={output.shape}, dtype={output.dtype}")
    return output

# Python 3 traceback module to identify which blocks you're in
import traceback
for line in traceback.format_stack()[-10:]:
    if "BasicTransformerBlock" in line:
        print(line.strip())
```

Remove debug hooks before finalizing — they add overhead and clutter the console.

## Common Pitfalls

**Pitfall 1: Not handling clonability.** If your patch stores state (caches, tensors), implement the clone callback to copy or reinitialize that state. Otherwise, `model.clone()` shares your state between all clones.

**Pitfall 2: Forward hook leaks.** Raw `register_forward_hook` hooks must be explicitly removed. If you register a hook and don't remove it, it runs forever — including during subsequent sampling calls from other graphs.

**Pitfall 3: Breaking gradient computation.** Even though ComfyUI runs in inference mode, some patches inadvertently trigger gradient bookkeeping by using operations that require gradients. Use `with torch.no_grad():` if your patch does anything beyond pure inference.

**Pitfall 4: Forgetting dtype casting.** If your patch introduces new tensors (cache entries, gate values), cast them to match the model's dtype: `torch.zeros_like(x, dtype=x.dtype)`, not `torch.zeros_like(x)`.

**Pitfall 5: Cache interference between cond and uncond.** During CFG, the model forward is called twice per step — once for cond, once for uncond. Your caching must distinguish between these two passes, either by checking `extra_options["cond_or_uncond"]` or by using separate cache keys.