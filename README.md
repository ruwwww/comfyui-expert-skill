# comfyui-core-node

Expert skill for developing ComfyUI custom nodes that touch ComfyUI's internal architecture — model management, attention patching, sampling pipeline, latent ops, text encoders, model architecture integration, and activation/dataflow hooks.

## What this covers

- Implementing research papers into ComfyUI (SAG, IP-Adapter, TeaCache, etc.)
- Wrapping new model architectures (DiT, Flux, GGUF)
- Custom samplers and attention processors
- Patching ComfyUI's forward pass via `transformer_options` / `model_options`
- Model management (clone, partial load, weight patching)
- Conditioning injection (ControlNet, IP-Adapter patterns)
- Latent and VAE operations

## Structure

```
├── SKILL.md                          # Main skill instructions (<500 lines)
├── references/
│   ├── node_registration.md          # NODE_CLASS_MAPPINGS, type contracts
│   ├── model_management.md           # VRAMState, ModelPatcher, clone, checkpoint loading
│   ├── attention_patching.md         # patches_replace, attn1/2_patch, extra_options
│   ├── sampling_pipeline.md          # KSampler, sample_custom, CFGGuider, sigmas
│   ├── latent_ops.md                 # LATENT dict contract, VAE encode/decode, masks
│   ├── conditioning.md               # CONDITIONING type, CLIP/T5 encoding, injection
│   ├── model_architecture.md         # supported_models, wrapping HF/diffusers, ModelType
│   └── activation_patching.md        # WrappersMP, CallbacksMP, transformer_options hooks
└── evals/
    └── evals.json                    # 3 benchmark test cases
```

## 10 Best Practices

| ID | Practice |
|----|----------|
| BP-01 | Read source code at implementation time; never trust training-time memory |
| BP-02 | Use `model.clone()` — never mutate the original model directly |
| BP-03 | Register nodes via `NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS` |
| BP-04 | Use `set_model_options_patch_replace` for attention patching |
| BP-05 | Manipulate conditioning via helper functions (`comfy.cond_helpers`) |
| BP-06 | Wrap new architectures through `ModelPatcher`, not standalone wrappers |
| BP-07 | Use `sample_custom()` and `KSampler` — don't bypass the sampling pipeline |
| BP-08 | Template typing: use ComfyUI's type system for RETURN_TYPES |
| BP-09 | Transformer hooks go through `transformer_options`, not monkey-patching |
| BP-10 | Manage VRAM with `model_management.unet_offload_device()` patterns |

## Benchmark

Iteration-1 results (3 evals × 2 configs each):

| Config | Pass Rate |
|--------|-----------|
| **With Skill** | **100%** ± 0% |
| Without Skill | 46% ± 7% |
| **Δ** | **+54%** |

## Reference Repos

These custom node repos demonstrate the patterns described by this skill (read at implementation time, not hardcoded):
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache)
- [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)