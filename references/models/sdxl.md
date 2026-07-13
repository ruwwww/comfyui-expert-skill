# SDXL Model Reference Guide

This reference document outlines the source code locations, architectural flows, and routing structures of the SDXL (Stable Diffusion XL) model inside ComfyUI.

---

## 📂 1. Source Code Locations & Class Registry

* **Model UNet:** [`comfy/ldm/modules/diffusionmodel.py`](file:///home/kuroko/ComfyUI/comfy/ldm/modules/diffusionmodel.py)
  * Class: `DiffusionModel` (implements the core 2D ResBlock and Cross-Attention UNet architecture).
* **Text Encoders:** [`comfy/sdxl_clip.py`](file:///home/kuroko/ComfyUI/comfy/sdxl_clip.py)
  * Class: `SDXLClipModel` (manages dual tokenization and joint embeddings).
* **Supported Configs:** [`comfy/supported_models.py`](file:///home/kuroko/ComfyUI/comfy/supported_models.py)
  * Class: `SDXL` (defines target resolution bounds, scaling vectors, and block layouts).

---

## 📐 2. Architectural Highlights & Mathematical Flow

SDXL is a latent U-Net model utilizing dual-cross-attention text conditioning and micro-conditioning vectors.

### A. Core Mathematical Concepts
1. **Dual Text Encoding:** Combines embeddings from CLIP ViT-L (768-dim) and OpenCLIP ViT-G (1280-dim):
   $$\mathbf{c}_{\text{cross}} = \text{concat}(\mathbf{c}_{\text{ViT-L}}, \mathbf{c}_{\text{OpenCLIP}})$$
   yielding a 2048-dim joint cross-attention context vector.
2. **Micro-Conditioning:** Concatenates original image size, crop coordinates, and target aesthetic score into a 6-element embedding vector $\mathbf{y}$:
   $$\mathbf{c}_{\text{vector}} = \text{MLP}(\mathbf{y})$$
   which is added to the time-step embedding to modulate the ResBlocks.

### B. Tensor Routing & Shape Transformations

```
Input Latents [B, 4, H/8, W/8] ──► Downsampling Blocks (ResBlocks + Cross-Attention)
                                           ▲
                                           │ (Cross-Attention Context [B, 77, 2048])
                                    CLIP ViT-L + OpenCLIP ViT-G Embeddings
                                           │
                                           ▼
                                     Middle Block
                                           │
                                           ▼
                                  Upsampling Blocks (ResBlocks + Cross-Attention)
                                           │
                                           ▼
                                 Predicted Noise Tensor [B, 4, H/8, W/8]
```

---

## 🧠 3. Step-by-Step Execution Sequence

1. **Dual Text Tokenization:**
   * Text inputs pass through CLIP ViT-L and OpenCLIP ViT-G to output sequence tokens of size $[B, 77, 768]$ and $[B, 77, 1280]$ respectively. These are concatenated along the feature dimension to form $[B, 77, 2048]$.
2. **Micro-Conditioning Vectors:**
   * Image resolution details (original size, crop size) are mapped via linear layers to size $[B, D_{\text{embed}}]$ and added to the time embedding.
3. **U-Net Feature Routing:**
   * The U-Net encoder downsamples the latents $[B, 4, H/8, W/8]$ through ResBlocks (performing spatial convolution) and cross-attention blocks (routing text token context).
4. **Decoder Synthesis:**
   * UpBlocks merge features from the DownBlocks via skip connections, reconstruct spatial dimensions, and project the output back to the latent noise space $[B, 4, H/8, W/8]$.
