import os
import sys
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Set environment variables
os.environ["MSLK_PYTHON_ONLY"] = "1"

# Register paths
sys.path.insert(0, "/home/kuroko/exp")
sys.path.insert(0, "/home/kuroko/exp/comfy-kitchen")

import comfy_kitchen
from comfy_kitchen.backends.cuda import quantize_convrot_w4a4_weight

def lloyd_max_1d(data, num_centroids=16, steps=10):
    centroids = torch.linspace(data.min(), data.max(), num_centroids, device=data.device)
    for _ in range(steps):
        dists = torch.abs(data.unsqueeze(-1) - centroids)
        assignments = torch.argmin(dists, dim=-1)
        for i in range(num_centroids):
            mask = (assignments == i)
            if mask.any():
                centroids[i] = data[mask].mean()
    return centroids.contiguous()

def quantize_orbitquant_and_fht(weight, num_centroids=16, convrot_groupsize=256):
    # FHT quantization of residual weight
    qweight, wscales = quantize_convrot_w4a4_weight(weight, convrot_groupsize=convrot_groupsize)
    return qweight.contiguous(), wscales.contiguous(), torch.zeros(1, device=weight.device) # centroid dummy

def should_skip(name, shape):
    if len(shape) != 2 or "weight" not in name:
        return True
    
    # Hard skip list for critical paths and attention projection modules
    skip_keywords = [
        "net.llm_adapter",    # Embedding
        "net.t_embedder",     # Time embedding
        "adaln_modulation",   # Modulation scaling
        "net.blocks.0.",      # Block 0 gateway
        "final_layer",        # Output layer
        "output_proj",        # Output projections
        "attn.q_proj",        # Self-Attention projections (very sensitive!)
        "attn.k_proj",
        "attn.v_proj",
        "attn.out_proj",
        "cross_attn.q_proj",  # Cross-Attention projections (very sensitive!)
        "cross_attn.k_proj",
        "cross_attn.v_proj",
        "cross_attn.out_proj",
    ]
    for kw in skip_keywords:
        if kw in name:
            return True
            
    if shape[-1] % 256 != 0:
        return True
        
    return False

def main():
    src_path = "/mnt/data/backup/diffusion_models/anima-base-v1.0.safetensors"
    dest_path = "/mnt/data/backup/diffusion_models/anima-base-v1.0-svd-orbitquant-w4a4.safetensors"
    
    print(f"Loading original model: {src_path}...")
    state_dict = {}
    with safe_open(src_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            
    print(f"Loaded {len(state_dict)} tensors.")
    
    quantized_state_dict = {}
    quantization_metadata = {"format_version": "1.0", "layers": {}}
    
    quantized_count = 0
    skipped_count = 0
    rank = 16
    
    print("\nStarting Hybrid SVD + OrbitQuant W4A4 quantization...")
    for k, v in state_dict.items():
        if should_skip(k, v.shape):
            quantized_state_dict[k] = v
            skipped_count += 1
        else:
            base_name = k.rsplit(".", 1)[0]
            
            # SVD outlier extraction
            w_float = v.cuda().float()
            U, S, V = torch.linalg.svd(w_float, full_matrices=False)
            U_r = U[:, :rank]
            S_r = torch.diag(S[:rank])
            V_r = V[:rank, :]
            
            proj_up = (U_r @ S_r).contiguous()
            proj_down = V_r.t().contiguous()
            
            svd_branch = proj_up @ proj_down.t()
            w_resid = w_float - svd_branch
            
            try:
                # Quantize the residual weight
                qweight, wscales, centroids = quantize_orbitquant_and_fht(w_resid, num_centroids=16)
                
                # Save packed representations and low-rank vectors back to CPU
                quantized_state_dict[k] = qweight.cpu()
                quantized_state_dict[f"{base_name}.weight_scale"] = wscales.cpu().to(torch.bfloat16)
                quantized_state_dict[f"{base_name}.weight_proj_down"] = proj_down.cpu().to(torch.bfloat16)
                quantized_state_dict[f"{base_name}.weight_proj_up"] = proj_up.cpu().to(torch.bfloat16)
                quantized_state_dict[f"{base_name}.weight_centroids"] = centroids.cpu().to(torch.bfloat16)
                
                # Register in comfyui metadata
                quantization_metadata["layers"][base_name] = {"format": "svd_orbitquant_w4a4"}
                quantized_count += 1
                
                if quantized_count % 20 == 0:
                    print(f"  Quantized {quantized_count} layers...")
            except Exception as e:
                print(f"  Failed to quantize {k} due to: {e}. Keeping in BF16.")
                quantized_state_dict[k] = v
                skipped_count += 1
                
    print(f"\nQuantization complete! Quantized: {quantized_count} layers, Kept in BF16: {skipped_count} layers.")
    
    metadata["_quantization_metadata"] = json.dumps(quantization_metadata)
    
    print(f"Saving quantized model to {dest_path}...")
    save_file(quantized_state_dict, dest_path, metadata=metadata)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
