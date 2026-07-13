# FLUX Model Reference Guide

This reference document outlines the source code locations, architectural flows, and routing structures of the FLUX model inside ComfyUI.

---

## 📂 1. Source Code Locations & Class Registry

* **Model Core:** [`comfy/ldm/flux/model.py`](file:///home/kuroko/ComfyUI/comfy/ldm/flux/model.py)
  * Class: `Flux` (manages parameter loading, embeddings, and block execution).
* **Layer Definitions:** [`comfy/ldm/flux/layers.py`](file:///home/kuroko/ComfyUI/comfy/ldm/flux/layers.py)
  * Classes: `DoubleStreamBlock` (handles parallel dual-stream sequences) and `SingleStreamBlock` (handles concatenated unified sequences).
* **Text Encoders:** [`comfy/text_encoders/flux.py`](file:///home/kuroko/ComfyUI/comfy/text_encoders/flux.py)
  * Class: `FluxClipModel` (manages CLIP and T5 tokenizers and embedding setups).

---

## 📐 2. Architectural Highlights & Mathematical Flow

FLUX is a Rectified Flow model utilizing a Double-Stream Multimodal Diffusion Transformer (MMDiT).

### A. Core Mathematical Concepts
1. **Flow Matching Trajectories:** Learns straight paths between noise and data, solving:
   $$\frac{d\mathbf{x}_t}{dt} = \mathbf{u}_t(\mathbf{x}_t)$$
2. **Double-Stream Modulated Attention:** In `DoubleStreamBlock`, image patches and text embeddings are transformed using distinct projection streams:
   $$\mathbf{q}_{\text{img}} = \mathbf{W}_{\text{q\_img}} \mathbf{x}_{\text{img}}, \quad \mathbf{q}_{\text{txt}} = \mathbf{W}_{\text{q\_txt}} \mathbf{x}_{\text{txt}}$$
   They then perform mutual information exchange in a shared cross-attention phase before being mapped back by separate MLPs.

### B. Tensor Routing & Shape Transformations

```
  Image Latents [B, S_img, D_hidden] ────┐          ┌─── Text Tokens [B, S_txt, D_hidden]
                                         │          │
                                         ▼          ▼
                                  ┌────────────────────┐
                                  │ DoubleStreamBlock  │ (Parallel pathways)
                                  └─────────┬──────────┘
                                            │
                                            ▼ Concatenate sequences
                                  ┌────────────────────┐
                                  │ SingleStreamBlock  │ (Shared pathways)
                                  └─────────┬──────────┘
                                            │
                                            ▼ Slice out & discard text
                                     [B, S_img, D_hidden]
                                            │
                                            ▼
                                   [B, S_img, C_out * P^2]
                                            │
                                            ▼ Unpatchify
                                      [B, C_out, H, W]
```

---

## 🧠 3. Step-by-Step Execution Sequence

1. **Embedding & Projecting Inputs:**
   * Image patches are projected via `self.img_in` to $[B, S_{\text{img}}, D]$.
   * Text sequences are projected via `self.txt_in` to $[B, S_{\text{txt}}, D]$.
2. **Positional Encoding Setup:**
   * Relative positional coordinate IDs are mapped via `self.pe_embedder` which implements the multi-axis sine/cosine embedding math.
3. **Double-Stream Processing (`double_blocks`):**
   * Tensors run through $N$ blocks of `DoubleStreamBlock`. Image and text components maintain individual channel scales and are normalized via separate modulation paths.
4. **Single-Stream Processing (`single_blocks`):**
   * Image and text tensors are concatenated into a joint array of shape $[B, (S_{\text{img}} + S_{\text{txt}}), D]$. They pass through $M$ single-stream blocks where attention operations treat them as a single token array.
5. **Output Projection:**
   * The text token segments are sliced off. The remaining visual sequence is projected via `self.final_layer` back to $[B, C \times P^2]$, and unpatchified to construct the final predicted latent noise state.
