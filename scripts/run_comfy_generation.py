import json
import urllib.request
import urllib.parse
import os
import time

def queue_prompt(prompt_dict, port=9741):
    payload = json.dumps({"prompt": prompt_dict}).encode('utf-8')
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/prompt",
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            return res.get("prompt_id")
    except Exception as e:
        print(f"Error queuing prompt: {e}")
        return None

def wait_for_image(filename_prefix, output_dir="/home/kuroko/ComfyUI/output", timeout=180):
    print(f"Waiting for image with prefix '{filename_prefix}' in {output_dir}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        for file in os.listdir(output_dir):
            if file.startswith(filename_prefix) and file.endswith(".png"):
                file_path = os.path.join(output_dir, file)
                print(f"Found generated image: {file_path}")
                return file_path
        time.sleep(2)
    print("Timeout waiting for image.")
    return None

def main():
    # Load the template prompt from ComfyUI_01843_.png
    with open("/home/kuroko/exp/last_prompt.json", "r") as f:
        prompt = json.load(f)
        
    # Crucial: Bypass TorchCompile node "165" because comfy-kitchen custom kernels
    # crash during torch.compile tracing due to raw DLPack/FakeTensor tracing constraints.
    # Connect SageAttention patcher "166" model input directly to UNETLoader "156:44".
    prompt["166"]["inputs"]["model"] = ["156:44", 0]
    
    # We want to run generation for:
    # 1. Original BF16 Anima model
    # 2. Hybrid SVD + OrbitQuant W4A4 Anima model
    runs = [
        {"model_name": "anima-base-v1.0.safetensors", "prefix": "test_anima_bf16"},
        {"model_name": "anima-base-v1.0-svd-orbitquant-w4a4.safetensors", "prefix": "test_anima_hybrid_svd_orbit"}
    ]
    
    # Let's clean up any existing test files in output folder to avoid false positives
    output_dir = "/home/kuroko/ComfyUI/output"
    for file in os.listdir(output_dir):
        if file.startswith("test_anima_") and file.endswith(".png"):
            os.remove(os.path.join(output_dir, file))
            print(f"Cleaned old test file: {file}")
            
    for run in runs:
        print(f"\nTriggering generation for: {run['model_name']}...")
        
        # Modify the prompt dictionary for this run
        # Node '156:44' is the UNETLoader
        prompt["156:44"]["inputs"]["unet_name"] = run["model_name"]
        
        # Node '157' is the SaveImage node
        prompt["157"]["inputs"]["filename_prefix"] = run["prefix"]
        
        # Queue the job in local ComfyUI
        prompt_id = queue_prompt(prompt)
        if not prompt_id:
            print("Failed to queue prompt. Is ComfyUI running?")
            continue
            
        print(f"Prompt queued successfully. Prompt ID: {prompt_id}")
        
        # Wait for the image to be written to disk
        image_path = wait_for_image(run["prefix"], output_dir=output_dir)
        if image_path:
            # Copy to exp directory for easy access
            dest = os.path.join("/home/kuroko/exp", os.path.basename(image_path))
            import shutil
            shutil.copy(image_path, dest)
            print(f"Copied test image to: {dest}")

if __name__ == "__main__":
    main()
