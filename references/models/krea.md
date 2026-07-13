# Krea / Krea2 Model Reference Guide

This reference document outlines the source code locations, architectural flows, and routing structures of the Krea / Krea2 latent consistency model inside ComfyUI.

---

## 📂 1. Source Code Locations & Class Registry

* **Model Base:** [`comfy/ldm/krea2/model.py`](file:///home/kuroko/ComfyUI/comfy/ldm/krea2/model.py)
  * Class: `Krea2` (implements the high-speed U-Net backbone mapping and timestep modulation).
* **Text Encoders:** [`comfy/text_encoders/krea2.py`](file:///home/kuroko/ComfyUI/comfy/text_encoders/krea2.py)
  * Class: `Krea2ClipModel`.

---

## 📐 2. Architectural Highlights & Mathematical Flow

Krea is a distilled consistency model designed to map noise inputs directly onto clean target images in a single step (or very few steps).

### A. Core Mathematical Concepts
1. **Consistency Mapping (LCM):** Learns a function $\boldsymbol{f}_\theta(\mathbf{x}_t, t)$ that predicts the solution to the probability flow ODE at $t=\epsilon$ directly:
   $$\boldsymbol{f}_\theta(\mathbf{x}_t, t) \approx \mathbf{x}_0$$
   Unlike standard models that require iterative denoising steps $t \to t-1$, consistency models map the coordinate space directly to origin points in a single forward pass.

### B. Tensor Routing & Shape Transformations

```
Input Latent [B, C_latent, H, W]
  │
  ▼
Timestep Embed + Text Condition [B, S_txt, D_cond]
  │
  ▼
Shallow U-Net Backbone ──► Predict Velocity Vectors
  │
  ▼
Direct Linear Interpolation
  │
  ▼
Estimated Target Latents [B, C_latent, H, W] (Direct x_0 projection)
```

---

## 🧠 3. Step-by-Step Execution Sequence

1. **Input Encoding:**
   * Receives latent states $[B, C, H, W]$ and tokenizes text embedding conditioning.
2. **Timestep Modulation:**
   * The current timestep $t$ is mapped to a sinusoidal embedding and joined with text conditioning to construct modulation keys.
3. **U-Net Feature Pass:**
   * The U-Net structures process spatial layers. ResBlocks and skip connections process the spatial features and project them directly to predict the trajectory velocity field.
4. **Boundary Condition Enforcement:**
   * Evaluates the consistency boundary conditions to map predicted velocities onto $x_0$ space, outputting the reconstructed latent structure.
