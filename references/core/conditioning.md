# Conditioning and Text Encoders

This covers how ComfyUI represents conditioning, how text encoders produce it, and how to manipulate conditioning for ControlNet, IP-Adapter, and custom conditioning nodes.

## The CONDITIONING Type

`CONDITIONING` is a list of `[tensor, metadata_dict]` pairs:

```python
conditioning = [
    [tensor_embeddings, {"pooled_output": pooled_tensor, ...}],
    # Multiple pairs for multiple conditionings
]
```

Each pair represents one conditioning source. Multiple pairs can coexist — they're typically concatenated by the sampling pipeline.

```
Conditioning type: list[tuple[Tensor, dict]]
├── pair 0: [embedding_tensor, metadata]
│   ├── tensor: shape [seq_len, embed_dim] — the text embeddings
│   └── metadata dict:
│       ├── "pooled_output": Tensor  — pooled CLIP embeddings, shape [batch, embed_dim]
│       ├── "control": ControlNet signal (injected by ControlNetApply)
│       ├── "gligen": GLIGEN bounding box data (injected by GLIGEN nodes)
│       └── custom keys — IP-Adapter, regional prompting, etc
├── pair 1: [embedding_tensor2, metadata2]
└── ...
```

### The embedding tensor

- For SD1.5/SDXL: `[77, 768]` or `[77, 2048]` — fixed-length token embeddings
- For T5-based models: variable length, shape `[seq_len, 4096]`
- Shape can vary across pairs — don't assume a fixed sequence length
- Zero-padded for empty prompts: `[[0, ..., 0], {"pooled_output": torch.zeros(...)}]`

### The metadata dict

Always present, even if empty. It travels alongside the embedding tensor through the pipeline. Keys:

| Key | Source | Purpose |
|-----|--------|---------|
| `pooled_output` | CLIP encoder | Global image representation for SDXL/Flux |
| `control` | ControlNetApply | ControlNet conditioning signal |
| `gligen` | GLIGEN nodes | Bounding-box–conditioned generation |
| `area` | ConditioningArea nodes | Spatial region for regional prompting |
| `strength` | ConditioningSet nodes | Per-conditioning strength multiplier |
| `mask` | ConditioningSetMask | Spatial mask for this conditioning |
| Custom keys | Your node / IP-Adapter | Custom conditioning data for attention patches |

## CLIP / Text Encoder

CLIP encoding in ComfyUI outputs `CONDITIONING`:

```python
# Inside CLIPTextEncode:
tokens = clip.tokenize(text)  # Tokenize
embeddings, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
conditioning = [[embeddings, {"pooled_output": pooled}]]
```

### Dual encoder models (T5 + CLIP)

Flux and SD3/SD4 use both CLIP-l and T5. The encoding produces two cond pairs:

```python
conditioning = [
    [clip_embeddings, {"pooled_output": pooled}],
    [t5_embeddings, {"pooled_output": pooled}],
]
```

The model's forward pass receives both and uses each appropriately.

### Writing a text encoder node

When creating a node that produces `CONDITIONING` from text (e.g., with a custom encoder), the output format is:

```python
def encode(self, clip, text, **kwargs):
    tokens = clip.tokenize(text)
    embeddings, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return ([[embeddings, {"pooled_output": pooled}]],)
```

## Conditioning Manipulation

### Combining conditioning

To merge multiple conditioning inputs (like `ConditioningCombine`):

```python
def combine(self, conditioning_a, conditioning_b):
    # Simple concatenation of condition pairs
    return (conditioning_a + conditioning_b,)
```

### Zeroing out conditioning

For "unconditional" paths or negative prompts:

```python
def zero_out(self, conditioning):
    result = []
    for emb, meta in conditioning:
        result.append([torch.zeros_like(emb), meta.copy()])
    return (result,)
```

### Timestep-based conditioning

For conditioning that varies over the denoising process:

```python
result = [[embeddings, {
    "pooled_output": pooled,
    "start_percent": 0.0,  # Active from start
    "end_percent": 0.5,    # Inactive after 50%
}]]
```

### Regional conditioning

For prompts that apply to spatial regions:

```python
result = [[embeddings, {
    "pooled_output": pooled,
    "area": (y_percent, x_percent, h_percent, w_percent),
    "strength": 1.0,
}]]
```

The sampler reads `area` and `strength` to spatially modulate conditioning application.

## ControlNet Injection

ControlNet produces conditioning signals that modify the UNet's intermediate features. The injection pattern:

1. ControlNetApply loads a control model
2. It encodes the control image (edge map, depth, pose, etc)
3. It adds a `"control"` key to the conditioning metadata
4. During sampling, the UNet reads `"control"` and applies it at each block

The control signal structure (in conditioning metadata):

```python
"control": control_output  # Applied internally by the UNet
"control_apply_to_uncond": True/False  # Whether to apply to uncond as well
```

Custom ControlNet-like nodes should follow this same metadata pattern — add a key to the conditioning metadata and have the model patch read it from `model_options`.

## IP-Adapter Pattern

IP-Adapter injects image-derived embeddings into cross-attention. The injection uses the conditioning metadata + attention patches:

1. IP-Adapter node takes an image and an IP-Adapter model
2. It computes image embeddings via the adapter
3. It stores these embeddings in `model.model_options["transformer_options"]`
4. An attention patch (registered via `set_model_options_patch_replace`) reads these embeddings during forward pass and injects them into cross-attention

If you're implementing a new conditioning injection method (style transfer, reference-based generation, etc), the IP-Adapter pattern is the reference: store your data in `transformer_options`, patch attention to read it, and thread it through via `model.model_options`.

## Common Pitfalls

**Pitfall 1: Forgetting to copy metadata.** When manipulating conditioning tensors, always copy the metadata dict:

```python
result = []
for emb, meta in conditioning:
    new_emb = my_transform(emb)
    result.append([new_emb, meta.copy()])  # Don't share dict references
```

**Pitfall 2: Batch dimension mismatch.** Conditioning is per-prompt. If you're processing with batch > 1, make sure the conditioning matches the batch size or is repeatable.

**Pitfall 3: Empty conditioning for uncond.** The negative conditioning for CFG must be non-empty — at minimum `[[torch.zeros_like(emb), meta.copy()]]`. An empty list `[]` causes sampler errors.

**Pitfall 4: Not reading /comfy/conds.py.** This file contains the core conditioning processing logic. Before writing any conditioning manipulation, grep and read this file to understand how ComfyUI processes conditioning during sampling.