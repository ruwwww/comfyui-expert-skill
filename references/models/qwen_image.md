# Qwen Image / Qwen2-VL Model Reference Guide

This reference document outlines the source code locations, architectural flows, and routing structures of the Qwen Image and Qwen2-VL vision-language models inside ComfyUI.

---

## 📂 1. Source Code Locations & Class Registry

* **Model Core:** [`comfy/ldm/qwen_image/model.py`](file:///home/kuroko/ComfyUI/comfy/ldm/qwen_image/model.py)
  * Class: `QwenImageModel` (defines the visual feature projection pathways).
* **Vision-Language Encoders:** [`comfy/text_encoders/qwen_image.py`](file:///home/kuroko/ComfyUI/comfy/text_encoders/qwen_image.py) & [`comfy/text_encoders/qwen_vl.py`](file:///home/kuroko/ComfyUI/comfy/text_encoders/qwen_vl.py)
  * Classes: `QwenVLModel`, `QwenImageTokenizer`.
* **Nodes Registry:** [`comfy_extras/nodes_qwen.py`](file:///home/kuroko/ComfyUI/comfy_extras/nodes_qwen.py) (defines node wrappers for inference routing).

---

## 📐 2. Architectural Highlights & Mathematical Flow

Qwen VL uses a decoupled vision-language architecture where static visual feature planes are mapped into autoregressive language embedding spaces.

### A. Core Mathematical Concepts
1. **Cross-Attention Resampler Bridge:** High-resolution spatial visual feature maps $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$ are compressed to a small set of queries $\mathbf{Q} \in \mathbb{R}^{N \times D}$ using a cross-attention resampler layer:
   $$\text{Attention}(\mathbf{Q}, \mathbf{F}, \mathbf{F}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{F}^T}{\sqrt{d_k}}\right) \mathbf{F}$$
   This reduces dense visual coordinate tokens down to a fixed set of $N$ representational query tokens (usually 256).

### B. Tensor Routing & Shape Transformations

```
Input Image [B, C, H, W]
  │
  ▼
Vision Encoder (SigLIP/ViT) ──► Spatial Feature Grid [B, (H/P * W/P), C_vis]
  │
  ▼
Cross-Attention Resampler  ──► Latent Visual Query Tokens [B, N_query, D_llm]
  │
  ▼ Concatenate with Text Embeddings
Token Concatenation ◄── Input Text Tokens [B, S_txt, D_llm]
  │
  ▼ Shape: [B, (N_query + S_txt), D_llm]
Autoregressive LLM Decoder
  │
  ▼
Sequence Tokens (Logits)
```

---

## 🧠 3. Step-by-Step Execution Sequence

1. **Visual Feature Extraction:**
   * The input image $[B, C, H, W]$ passes through the vision encoder, returning spatial feature tokens of shape $[B, S_{\text{vis}}, C_{\text{vis}}]$.
2. **Cross-Attention Compression:**
   * A learnable query tensor $[B, N_{\text{query}}, D_{\text{llm}}]$ performs cross-attention over the spatial visual features to project them down to size $[B, N_{\text{query}}, D_{\text{llm}}]$.
3. **Sequence Interleaving:**
   * The text prompt instruction tokens are mapped to embeddings of shape $[B, S_{\text{txt}}, D_{\text{llm}}]$.
   * The visual queries are dynamic tokens interleaved at the exact position tags (`<image>`) inside the text sequence, building a unified token sequence.
4. **Autoregressive Generation:**
   * The merged token sequence passes through the language decoder (typically Qwen2/Qwen2.5) to autoregressively output logit arrays representing text tokens.
