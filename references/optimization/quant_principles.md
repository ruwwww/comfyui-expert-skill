# Mathematical & Engineering Principles of Quantization

This reference document details the core mathematical principles, attention rules, and serialization constraints used to quantize diffusion models.

---

## 📐 1. Mathematical Principles of Quantization

### A. FP8 Formats (E4M3 vs. E5M2)
* **E4M3 (1 Sign, 4 Exponent, 3 Mantissa):** Bounded at $\pm 240.0$. Has higher precision (3-bit mantissa) but lower dynamic range. **Preferred for weights and activations** during forward passes because representation accuracy is critical.
* **E5M2 (1 Sign, 5 Exponent, 2 Mantissa):** Bounded at $\pm 57344.0$. Identical dynamic range to FP16 but lower precision (2-bit mantissa). **Preferred for gradients and backward passes** where scale variations are extreme.

### B. SVDQuant (Bypassing Outlier Channels)
Large language and diffusion models develop highly concentrated **activation outliers** (specific channels with values 20x higher than the mean). Linear quantization forces the scaling factor to expand to capture these outliers, which collapses all normal values to zero.
* **Mathematical Separation:** SVDQuant factorizes the weight matrix $W$ into a low-bit quantized part and a high-precision low-rank outlier correction:
  $$W \approx W_{\text{quant}} + A B^T$$
* **Low-Rank Path:** The top-R singular dimensions ($A \in \mathbb{R}^{O \times R}, B \in \mathbb{R}^{I \times R}$) are extracted via Singular Value Decomposition (SVD).
* **Residual Path:** The remaining outlier-free matrix $W_{\text{residual}} = W - A B^T$ is quantized cleanly to 4-bit without scale expansion.

### C. OrbitQuant (Vector Quantization on Spheres)
Instead of quantizing individual scalars, OrbitQuant operates on high-dimensional vectors (rows of weight matrices).
* **Spherical Projection:** Rows are projected onto a unit sphere $\mathbb{S}^{N-1}$ by factoring out their L2 norm:
  $$W_{\text{row}} = \|W_{\text{row}}\|_2 \cdot \vec{v}$$
* **Concentration of Measure:** In high dimensions, random vectors on a sphere concentrate uniformly near the equator. Individual coordinate outliers are mathematically diluted.
* **Non-linear Centroids:** A lookup table (LUT) of centroids is optimized via a 1D Lloyd-Max density-fitting algorithm to fit the spherical coordinates, achieving superior fidelity over uniform linear grids.

### D. ConvRot (Walsh-Hadamard orthogonal rotation)
* **Outlier Dilution:** Multiplies weights and activations by an orthogonal Walsh-Hadamard transform matrix $R$ ($R^T R = I$).
* **Mechanism:** Spreads the energy of outlier spikes across all $N$ channels by scaling them by $1/\sqrt{N}$. This flattens the activation distribution, allowing standard 4-bit uniform quantizers to function without clipping.

---

## 🧠 2. The Attention Quantization Rule
* **The Softmax Sensitivity:** Self-attention and cross-attention projection layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`) are highly sensitive to low-bit quantization.
* **Mechanism:** Attention weights are calculated via:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  Because the softmax function contains an exponential component ($e^x$), even microscopic quantization noise in the query-key projections is scaled exponentially, resulting in complete structural degradation of the generated images.
* **Engineering Best Practice:** Always preserve attention projections in **BF16/FP16** and focus W4A4/W8A8 quantization exclusively on the large MLP/FFN expansion layers (which occupy ~70% of the model parameters).

---

## 💾 3. Layout Serialization Constraints (`state_dict_tensors`)
When ComfyUI duplicates or offloads modules, it relies on `module.state_dict()`. Custom layouts **must** define the serialization mapping back to disk-format suffixes:
```python
@classmethod
def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
    return {
        "": qdata,                           # Maps to standard *.weight key
        "_scale": params.scale,              # Maps to *.weight_scale
        "_proj_down": params.proj_down,      # Maps to *.weight_proj_down
        "_proj_up": params.proj_up,          # Maps to *.weight_proj_up
        "_centroids": params.centroids,      # Maps to *.weight_centroids
    }
```
