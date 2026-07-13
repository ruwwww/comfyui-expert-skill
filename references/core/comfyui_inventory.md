# ComfyUI Inventory Skill Reference Guide

This reference document outlines the discovery modes, cache schema, workflow validation rules, and package mappings for the ComfyUI Inventory system.

---

## 📋 1. Purpose & Workflow Validation

Every workflow generation MUST be preceded by an inventory check. This prevents:
* Referencing models that aren't downloaded.
* Using nodes that aren't installed.
* Exceeding GPU VRAM physical limits.

---

## 🔌 2. Two Discovery Modes

### A. Online Mode (ComfyUI API Running)
Query the live server for authoritative information.

1. **System Info:** `GET http://127.0.0.1:8188/system_stats`
   * Extracts: GPU name, total VRAM, free VRAM, and ComfyUI version.
2. **Installed Nodes:** `GET http://127.0.0.1:8188/object_info`
   * Returns all registered node classes with their input/output specifications.
3. **Installed Models:** Query model-type endpoints:
   * `/models/checkpoints`
   * `/models/loras`
   * `/models/vae`
   * `/models/controlnet`
   * `/models/clip`
   * `/models/clip_vision`
   * `/models/upscale_models`
   * `/models/diffusion_models`

### B. Offline Mode (Directory Scan)
When ComfyUI is not running, scan the filesystem directories directly under the `{ComfyUI}` installation root:

* `{ComfyUI}/models/checkpoints/` $\to$ `.safetensors`, `.ckpt`
* `{ComfyUI}/models/loras/` $\to$ `.safetensors`
* `{ComfyUI}/models/vae/` $\to$ `.safetensors`, `.pt`
* `{ComfyUI}/models/controlnet/` $\to$ `.safetensors`, `.pth`
* `{ComfyUI}/models/clip/` $\to$ `.safetensors`
* `{ComfyUI}/models/clip_vision/` $\to$ `.safetensors`
* `{ComfyUI}/models/upscale_models/` $\to$ `.pth`, `.safetensors`
* `{ComfyUI}/models/diffusion_models/` $\to$ `.safetensors`
* `{ComfyUI}/models/ipadapter/` $\to$ `.safetensors`, `.bin`
* `{ComfyUI}/models/instantid/` $\to$ `.bin`
* `{ComfyUI}/models/insightface/` $\to$ `.onnx`
* `{ComfyUI}/models/facerestore_models/` $\to$ `.pth`
* `{ComfyUI}/models/ultralytics/bbox/` $\to$ `.pt`
* `{ComfyUI}/custom_nodes/` $\to$ Folder names correspond to node packages (e.g., `ComfyUI-Impact-Pack`).

---

## 💾 3. Cache Format (`state/inventory.json`)

The inventory outputs a serialized JSON cache to `state/inventory.json`:

```json
{
  "last_updated": "2026-02-06T12:00:00Z",
  "mode": "online",
  "comfyui_version": "0.3.10",
  "system": {
    "gpu": "NVIDIA RTX 5090",
    "vram_total_gb": 32,
    "vram_free_gb": 28
  },
  "models": {
    "checkpoints": ["flux1-dev.safetensors", "RealVisXL_V5.0.safetensors"],
    "loras": ["sage_character.safetensors"],
    "vae": ["ae.safetensors", "wan_2.1_vae.safetensors"],
    "controlnet": ["instantid_controlnet.safetensors"],
    "clip": ["t5xxl_fp16.safetensors", "clip_l.safetensors"],
    "clip_vision": ["CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"],
    "upscale_models": ["4x-UltraSharp.pth"],
    "diffusion_models": ["wan2.1_i2v_720p_14b_bf16.safetensors"],
    "ipadapter": ["ip-adapter-faceid-plusv2_sd15.bin"],
    "instantid": ["ip-adapter.bin"],
    "insightface": ["inswapper_128.onnx"],
    "facerestore": ["codeformer.pth"],
    "detection": ["face_yolov8m.pt"]
  },
  "custom_nodes": [
    "ComfyUI-Manager",
    "ComfyUI_IPAdapter_plus",
    "ComfyUI-Impact-Pack"
  ]
}
```

---

## 🛠️ 4. Validation Rules & Node-to-Package Mapping

Given a target workflow JSON, validate against the active inventory:

1. **For each node:** Check `class_type` against known classes. If missing, identify the custom node package and output:
   * `"Install via ComfyUI-Manager: {package_name}"`
2. **For each model reference:** Check filename. If missing, retrieve the target download URL from registry files and output:
   * `"Missing: {filename} - Download from {url} -> {path}"`

### 🗺️ Common Node Class Mappings
| Node Class | Target Package |
| :--- | :--- |
| `ApplyInstantID` | `ComfyUI_InstantID` |
| `IPAdapterUnifiedLoader` | `ComfyUI_IPAdapter_plus` |
| `FaceDetailer` | `ComfyUI-Impact-Pack` |
| `ReactorFaceSwap` | `ComfyUI-ReActor` |
| `AnimateDiffLoaderWithContext` | `ComfyUI-AnimateDiff-Evolved` |
| `VideoHelper*` / `VHS_*` | `ComfyUI-VideoHelperSuite` |
| `ControlNetApply*` | `comfyui_controlnet_aux` |
| `UltimateSDUpscale` | `ComfyUI_UltimateSDUpscale` |
| `RIFE*` | `ComfyUI-Frame-Interpolation` |

---

## ⚡ 5. Integration & Cache Freshness

* **Validity Bounds:** The cache is valid for **1 hour** during active sessions.
* **Invalidation Trigger:** Invalidate and force refresh when the user installs new models or packages.
* **Integration Points:**
  * Called by `comfyui-workflow-builder` before generating workflows.
  * Called by `comfyui-character-gen` for model selection.
  * Called by `comfyui-troubleshooter` when diagnosing execution failures.
