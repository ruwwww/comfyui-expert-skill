# API & Workflow Integration for Custom Nodes

This reference guide details how custom node developers interface their nodes with ComfyUI's REST API (`comfyui-api` skill) and the natural language workflow generation engine (`comfyui-workflow-builder` skill) for general development and testing.

---

## 🔌 1. API Discovery & Node Registration

For custom nodes to be automatically discovered by external APIs and workflow builders:

1. **`object_info` Registration:** ComfyUI exposes all registered nodes and their input/output type signatures via the `GET /object_info` endpoint.
2. **Declaring Parameter Signatures:** ComfyUI derives the API schema directly from your node's `INPUT_TYPES()` classmethod. Ensure parameter types are strictly declared to allow schema validation:
   ```python
   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {
               "image": ("IMAGE",),
               "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
           }
       }
   ```
3. **Inventory Caching:** Client-side builders scan this metadata to cache the node signatures in `state/inventory.json`, permitting automated graph validation before queuing runs.

---

## 🎨 2. Workflow JSON Structure

When programmatically assembling workflow graphs to trigger your custom node via the API:

* **Node Connection Schema:** Interconnect node outputs and inputs using the standard string-keyed JSON dict format. Output links are defined as a list `[source_node_id, output_index]`:
  ```json
  {
    "10": {
      "class_type": "MyCustomUpscalerNode",
      "inputs": {
        "image": ["9", 0],
        "upscale_factor": 2.0
      }
    }
  }
  ```

---

## ⚡ 3. API Execution & Benchmarking Loop

To verify your custom node execution output and gather node-level processing benchmarks:

1. **Queue Graph Execution:** Send a `POST` request to `/prompt` with the workflow JSON:
   ```bash
   curl -X POST http://127.0.0.1:8188/prompt \
     -H "Content-Type: application/json" \
     -d '{"prompt": WORKFLOW_JSON, "client_id": "development-client"}'
   ```
2. **Monitor Execution History:** Poll the `/history/{prompt_id}` endpoint. Once the status shows `completed: true`, retrieve the output keys and check the detailed node-by-node execution durations inside the messages log:
   ```json
   {
     "prompt_id-abc": {
       "outputs": {
         "10": {
           "images": [{"filename": "output_0001.png", "type": "output"}]
         }
       },
       "status": {
         "completed": true,
         "messages": [
           ["execution_success", {"node_id": "10", "execution_time": 0.45}]
         ]
       }
     }
   }
   ```
