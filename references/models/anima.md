# Anima Model Reference Guide

This reference document outlines the source code locations, architectural flows, and routing structures of the Anima (Cosmos / WanVAE Core) model inside ComfyUI.

---

## 📂 1. Source Code Locations & Class Registry

* **Model Backbone:** [`comfy/ldm/cosmos/predict2.py`](file:///home/kuroko/ComfyUI/comfy/ldm/cosmos/predict2.py)
  * Main class: `MiniTrainDIT` (implements the 3D-attention sequence layers and patch projection).
* **Anima Wrapper:** [`comfy/ldm/anima/model.py`](file:///home/kuroko/ComfyUI/comfy/ldm/anima/model.py)
  * Implements `Attention`, `RotaryEmbedding`, and `TransformerBlock` mapping wrapper layers.
* **Text Encoders:** [`comfy/text_encoders/anima.py`](file:///home/kuroko/ComfyUI/comfy/text_encoders/anima.py)
  * Class: `AnimaModel` (handles dual language text embeddings).

---

## 📐 2. Architectural Highlights & Mathematical Flow

Anima is a dense flow-matching Diffusion Transformer (DiT) utilizing a single-stream joint attention mechanism.

### A. Core Mathematical Concepts
1. **Flow Matching:** Operates on continuous probability trajectories instead of standard discrete noise schedules, estimating velocity vector fields $\mathbf{v}_t$ directly:
   $$\mathbf{v}_t = \frac{d\mathbf{x}_t}{dt}$$
2. **Rotary Position Embeddings (3D-RoPE):** Applies positional coordinates across height, width, and temporal sequences:
   $$\mathbf{x}_{\text{embed}} = (\mathbf{x} \odot \cos\Theta) + (\text{rotate\_half}(\mathbf{x}) \odot \sin\Theta)$$
   * Implemented in [`comfy/ldm/anima/model.py` at line 13](file:///home/kuroko/ComfyUI/comfy/ldm/anima/model.py#L13).

### B. Tensor Routing & Shape Transformations

```
Input Latents [B, C_latent, T, H, W]
  │
  ▼ Rearrange Space/Time (Patchify)
Patch Embedder ──► [B, S_img, D_hidden]
  │
  ▼ Concat along sequence dimension
Token Mixer ◄── Text Tokens [B, S_txt, D_hidden]
  │
  ▼ Shape: [B, (S_img + S_txt), D_hidden]
Transformer Blocks (1..N)
  │
  ▼ Split Sequence & Discard Text
Split Tokens ──► [B, S_img, D_hidden]
  │
  ▼ Linear Projection & Reshape
Final Output ──► Noise Velocity Estimate [B, C_latent, T, H, W]
```

---

## 🧠 3. Step-by-Step Execution Sequence

1. **Patchification & Projection:** 
   The input latent tensor $[B, C, T, H, W]$ is patchified (using standard patch size $P=2$) and projected to a flat token representation $[B, S_{\text{img}}, D]$ where $S_{\text{img}} = T \times \frac{H}{P} \times \frac{W}{P}$.
2. **Text Conditioning Injection:**
   The text encoder yields context embeddings of shape $[B, S_{\text{txt}}, D]$. These are concatenated with the image tokens to form a single sequence of shape $[B, (S_{\text{img}} + S_{\text{txt}}), D]$.
3. **Joint Attention Blocks:**
   The joint sequence passes through $N$ transformer blocks. In each block, self-attention allows visual patches to directly attend to text tokens and other visual patches without cross-attention projection overhead.
4. **Unpatchification:**
   The text tokens are sliced out and discarded. The remaining visual patches of shape $[B, S_{\text{img}}, D]$ are linearly projected back to channel dimension $C \times P^2$ and reshaped to rebuild the latent output shape $[B, C, T, H, W]$.
