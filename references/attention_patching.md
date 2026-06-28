# Attention Mechanisms and Patching

This covers how to modify attention computation in ComfyUI models — the core technique behind IP-Adapter, Self-Attention Guidance, attention masking, and most paper implementations involving attention.

## The ComfyUI Attention System

All attention in ComfyUI flows through a centralized abstraction in `/comfy/ldm/modules/attention.py`. The key design decision: ComfyUI's transformer blocks don't compute attention directly — they delegate to configurable attention functions stored in `model_options["transformer_options"]`.

### BasicTransformerBlock

In `/comfy/ldm/modules/attention.py`, the `BasicTransformerBlock` class handles both self-attention (`attn1`) and cross-attention (`attn2`):

```python
# Simplified from attention.py:
class BasicTransformerBlock(nn.Module):
    def forward(self, x, context=None, transformer_options={}):
        # Self-attention
        n = transformer_options.get("block", (None, None, None))[1]  # block index
        q, k, v = self.attn1.to_q(x), self.attn1.to_k(x), self.attn1.to_v(x)
        
        # Look for a registered patch for attn1 at this block
        attn1_patch = transformer_options.get("patches_replace", {}).get("attn1", {}).get(("input", n))
        if attn1_patch:
            x = attn1_patch(q, k, v, transformer_options)
        else:
            x = self.attn1(q, k, v, transformer_options)
        
        # Cross-attention (only if we have context)
        if context is not None:
            attn2_patch = transformer_options.get("patches_replace", {}).get("attn2", {}).get(("input", n))
            if attn2_patch:
                # Custom attention! Your patch runs here
                x = attn2_patch(q, k, v, transformer_options)
            else:
                x = self.attn2(q, k, v, transformer_options)
```

The exact hook points vary by model architecture (DiT models like Flux have different block structures). Always read the actual forward pass of the model you're patching — grep for `patches_replace` in `/comfy/` to find all hook sites.

## Patching Attention

### Method 1: patches_replace (recommended for surgical patches)

Use `set_model_options_patch_replace` to replace attention computation at specific blocks:

```python
from comfy.model_patcher import set_model_options_patch_replace

def my_attn2_patch(n, context_attn2, original_attn2=None):
    """Called during sampling for each transformer block n.
    
    Args:
        n: block index
        context_attn2: current cross-attention context  
    
    Returns:
        New context_attn2 value
    """
    # Manipulate attention here
    return context_attn2

# For targeted attention replacement:
from comfy.ldm.modules.attention import optimized_attention

class MyCrossAttentionReplace:
    def __call__(self, q, k, v, extra_options):
        # q, k, v: tensors for this attention call
        # extra_options: contains "n_heads", "block", "transformer_index", etc
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        # Modify out here...
        return out

    def to(self, device, dtype=None):
        # If your patch stores tensors, move them to device here
        return self

model = model.clone()
for n in range(num_blocks):
    model_options = set_model_options_patch_replace(
        model.model_options, 
        MyCrossAttentionReplace(), 
        "attn2",         # target: "attn1" (self) or "attn2" (cross)
        "input",         # block_name: typically "input", "middle", "output"
        n                # block number
    )
    model.model_options = model_options
```

### Method 2: set_model_attn1_patch / set_model_attn2_patch (for global attention hooks)

These are simpler but less granular — they apply a function that runs for *every* block:

```python
model = model.clone()

def attention_hook(n, context_attn2=0):
    # n: block number
    # context_attn2: the cross-attention context for block n
    # Return modified context_attn2 (or just the original)
    return context_attn2

model.set_model_attn1_patch(attention_hook)  # Self-attention hook
model.set_model_attn2_patch(attention_hook)  # Cross-attention hook
```

This is the pattern used by IP-Adapter's attention injection — they read `extra_options["block"]` to determine which block they're in and selectively modify attention.

## Cross-Attention vs Self-Attention

- **Self-attention (attn1)**: Query, key, and value all come from the latent representation. Controls how spatial regions relate to each other.
- **Cross-attention (attn2)**: Query from latent, key/value from text/conditioning embeddings. Controls how the image attends to the prompt.

Most paper implementations modify cross-attention (to inject styles, control which parts of the prompt affect which regions). Self-attention modifications are rarer but important for things like Self-Attention Guidance (SAG) and attention-based upscaling.

## The extra_options dict

When your custom attention function (or patches_replace function) runs, it receives `extra_options` with these keys:

| Key | Description |
|-----|-------------|
| `n_heads` | Number of attention heads |
| `n_repeats` | Number of times the attention is repeated |
| `block` | `(block_name, block_index, transformer_index)` — identifies which block this is |
| `transformer_index` | For dual-transformer models, which transformer stream |
| `original_shape` | Original spatial shape before flattening |
| `cond_or_uncond` | Mask indicating which batch items are conditioned vs unconditional |
| `sigmas` | Current noise level (tensor) — useful for sigma-dependent attention |
| `ad_params` | AnimateDiff parameters (if AD is active) |

## Pitfalls

**Pitfall 1: Breaking xformers/flash-attn compatibility.** When replacing `optimized_attention`, you lose the automatic selection of the fastest backend. If you need custom logic around attention but don't need to replace the attention itself, wrap rather than replace:

```python
class Attn2Wrapper:
    def __call__(self, q, k, v, extra_options):
        # Pre-process q (e.g., inject conditioning into k/v)
        k = my_modify_k(k, extra_options)
        v = my_modify_v(v, extra_options)
        # Still use the optimized path
        return optimized_attention(q, k, v, extra_options["n_heads"])
```

**Pitfall 2: Ignoring sigma ranges.** Many attention patches should only activate during specific denoising stages (early = structure, late = details). Check `extra_options["sigmas"]` and gate your patch:

```python
if extra_options["sigmas"].mean() > sigma_threshold:
    return original_path(q, k, v, extra_options)  # Early steps: skip patch
# Late steps: apply patch
```

**Pitfall 3: Not handling cond/uncond split.** During CFG, the batch is split into conditioned and unconditional halves. Your attention patch may see batches where half the items need the patch and half don't:

```python
cond_or_uncond = extra_options["cond_or_uncond"]
cond_mask = cond_or_uncond[:len(cond_or_uncond)//2]  # First half
# Apply patch differently for cond vs uncond
```

**Pitfall 4: Device mismatch.** Custom tensors in your patch (e.g., IP-Adapter image embeddings) must move to the same device as the attention computation. Use `.to(q.device)` or `.to(dtype=q.dtype)` inside the `__call__` method rather than pre-computing on a fixed device. The `to()` method on your replacement class is called by ComfyUI during model loading — implement it to move your stored tensors.