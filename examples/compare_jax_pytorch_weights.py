#!/usr/bin/env python3
"""
Compare JAX and PyTorch model weights to identify conversion issues.

Usage:
    python examples/compare_jax_pytorch_weights.py \
        --jax_checkpoint_dir /path/to/jax/checkpoint \
        --pytorch_checkpoint_dir /path/to/pytorch/checkpoint \
        --config_name pi05_b1k_inference_final
"""

import os
import json
import numpy as np
import torch
import tyro
from typing import Literal
from safetensors import safe_open

from flax.nnx import traversals
import openpi.models.model
import openpi.models.pi0_config
import openpi.training.config as _config


def load_jax_params(checkpoint_dir: str):
    """Load JAX checkpoint parameters."""
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype="float32"
    )
    return params


def load_pytorch_params(checkpoint_dir: str):
    """Load PyTorch checkpoint parameters."""
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert bfloat16 to float32 for numpy compatibility
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            tensors[key] = tensor.numpy()
    return tensors


def compare_weights(
    jax_checkpoint_dir: str,
    pytorch_checkpoint_dir: str,
    config_name: str,
    verbose: bool = False,
):
    """Compare JAX and PyTorch weights."""
    print(f"Loading JAX checkpoint from {jax_checkpoint_dir}")
    jax_params = load_jax_params(jax_checkpoint_dir)
    
    print(f"Loading PyTorch checkpoint from {pytorch_checkpoint_dir}")
    pytorch_params = load_pytorch_params(pytorch_checkpoint_dir)
    
    model_config = _config.get_config(config_name).model
    print(f"Model config: pi05={model_config.pi05}, num_tasks={model_config.num_tasks}")
    
    print("\n" + "="*80)
    print("COMPARING PROJECTION LAYERS (action_in_proj, action_out_proj, time_mlp, etc.)")
    print("="*80)
    
    if model_config.pi05:
        proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        proj_keys = ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]
    
    for key in proj_keys:
        print(f"\n--- {key} ---")
        
        # JAX params
        jax_kernel = jax_params[key]["kernel"]
        jax_bias = jax_params[key]["bias"]
        if isinstance(jax_kernel, dict):
            jax_kernel = jax_kernel["value"]
            jax_bias = jax_bias["value"]
        
        # PyTorch params
        pt_weight = pytorch_params.get(f"{key}.weight")
        pt_bias = pytorch_params.get(f"{key}.bias")
        
        if pt_weight is None:
            print(f"  ERROR: {key}.weight not found in PyTorch checkpoint!")
            continue
        
        # JAX kernel shape is (in_features, out_features)
        # PyTorch weight shape is (out_features, in_features) - transposed
        jax_kernel_t = jax_kernel.T
        
        print(f"  JAX kernel shape: {jax_kernel.shape} -> transposed: {jax_kernel_t.shape}")
        print(f"  PyTorch weight shape: {pt_weight.shape}")
        
        if jax_kernel_t.shape != pt_weight.shape:
            print(f"  SHAPE MISMATCH!")
        else:
            diff = np.abs(jax_kernel_t - pt_weight)
            print(f"  Weight max diff: {diff.max():.6e}, mean diff: {diff.mean():.6e}")
            
        if pt_bias is not None:
            diff_bias = np.abs(jax_bias - pt_bias)
            print(f"  Bias max diff: {diff_bias.max():.6e}, mean diff: {diff_bias.mean():.6e}")
    
    # Check task_embeddings if present
    if model_config.num_tasks > 0:
        print("\n" + "="*80)
        print("COMPARING TASK EMBEDDINGS")
        print("="*80)
        
        # JAX task embeddings
        jax_task_emb = jax_params.get("task_embeddings")
        if jax_task_emb is not None:
            if "embedding" in jax_task_emb:
                jax_emb_weight = jax_task_emb["embedding"]
            elif "kernel" in jax_task_emb:
                jax_emb_weight = jax_task_emb["kernel"]
            else:
                print(f"  JAX task_embeddings structure: {jax_task_emb.keys() if isinstance(jax_task_emb, dict) else type(jax_task_emb)}")
                jax_emb_weight = None
            
            if isinstance(jax_emb_weight, dict) and "value" in jax_emb_weight:
                jax_emb_weight = jax_emb_weight["value"]
            
            print(f"  JAX task_embeddings keys: {jax_task_emb.keys() if isinstance(jax_task_emb, dict) else 'N/A'}")
            if jax_emb_weight is not None:
                print(f"  JAX embedding shape: {jax_emb_weight.shape}")
        else:
            print("  JAX task_embeddings not found!")
            jax_emb_weight = None
        
        # PyTorch task embeddings
        pt_task_weight = pytorch_params.get("task_embeddings.weight")
        if pt_task_weight is not None:
            print(f"  PyTorch embedding shape: {pt_task_weight.shape}")
            
            if jax_emb_weight is not None:
                # Embeddings should match directly (no transpose needed)
                if jax_emb_weight.shape != pt_task_weight.shape:
                    print(f"  SHAPE MISMATCH! JAX: {jax_emb_weight.shape}, PyTorch: {pt_task_weight.shape}")
                    # Try transposed
                    if jax_emb_weight.T.shape == pt_task_weight.shape:
                        print(f"  NOTE: Shapes match if JAX is transposed!")
                        diff = np.abs(jax_emb_weight.T - pt_task_weight)
                        print(f"  If transposed - max diff: {diff.max():.6e}, mean diff: {diff.mean():.6e}")
                else:
                    diff = np.abs(jax_emb_weight - pt_task_weight)
                    print(f"  Embedding max diff: {diff.max():.6e}, mean diff: {diff.mean():.6e}")
        else:
            print("  PyTorch task_embeddings.weight not found!")
            # Check what keys exist
            task_keys = [k for k in pytorch_params.keys() if "task" in k.lower()]
            print(f"  Available task-related keys: {task_keys}")
    
    # Check a few Gemma expert layers
    print("\n" + "="*80)
    print("COMPARING GEMMA EXPERT ATTENTION WEIGHTS (layer 0)")
    print("="*80)
    
    # Flatten JAX PaliGemma params
    paligemma_flat = traversals.flatten_mapping(jax_params["PaliGemma"], sep="/")
    
    # Check for expert attention weights
    suffix = "/value" if "llm/layers/attn/q_einsum_1/w/value" in paligemma_flat else ""
    
    # Q projection
    jax_q = paligemma_flat.get(f"llm/layers/attn/q_einsum_1/w{suffix}")
    if jax_q is not None:
        print(f"\n  JAX q_einsum_1 shape: {jax_q.shape}")
        
        # Convert to PyTorch format
        # JAX shape: (num_layers, num_heads, hidden_size, head_dim)
        # PyTorch shape: (num_heads * head_dim, hidden_size)
        layer_0_q = jax_q[0]
        print(f"  JAX layer 0 q shape: {layer_0_q.shape}")
        
        # Correct conversion (matches convert_jax_model_to_pytorch.py):
        # transpose(0, 2, 1): (num_heads, hidden_size, head_dim) -> (num_heads, head_dim, hidden_size)
        # reshape: (num_heads * head_dim, hidden_size)
        num_heads = layer_0_q.shape[0]
        head_dim = layer_0_q.shape[2]
        hidden_size = layer_0_q.shape[1]
        jax_q_converted = layer_0_q.transpose(0, 2, 1).reshape(num_heads * head_dim, hidden_size)
        print(f"  JAX q converted shape: {jax_q_converted.shape}")
        
        pt_q = pytorch_params.get("paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight")
        if pt_q is not None:
            print(f"  PyTorch q_proj shape: {pt_q.shape}")
            if jax_q_converted.shape == pt_q.shape:
                diff = np.abs(jax_q_converted - pt_q)
                print(f"  Q proj max diff: {diff.max():.6e}, mean diff: {diff.mean():.6e}")
            else:
                print(f"  SHAPE MISMATCH! JAX: {jax_q_converted.shape}, PyTorch: {pt_q.shape}")
    
    # O projection - this might have a bug in conversion
    jax_o = paligemma_flat.get(f"llm/layers/attn/attn_vec_einsum_1/w{suffix}")
    if jax_o is not None:
        print(f"\n  JAX attn_vec_einsum_1 shape: {jax_o.shape}")
        layer_0_o = jax_o[0]
        print(f"  JAX layer 0 o shape: {layer_0_o.shape}")
        
        # Original conversion (in slice_gemma_state_dict):
        # .reshape(num_heads * head_dim, hidden_size).transpose(1, 0)
        jax_o_converted_original = layer_0_o.reshape(-1, layer_0_o.shape[-1]).transpose(1, 0)
        print(f"  JAX o converted (original method) shape: {jax_o_converted_original.shape}")
        
        # PaliGemma conversion method:
        # .transpose(2, 0, 1).reshape(...)
        jax_o_converted_pali = layer_0_o.transpose(2, 0, 1).reshape(-1, layer_0_o.shape[-1])
        print(f"  JAX o converted (pali method) shape: {jax_o_converted_pali.shape}")
        
        pt_o = pytorch_params.get("paligemma_with_expert.gemma_expert.model.layers.0.self_attn.o_proj.weight")
        if pt_o is not None:
            print(f"  PyTorch o_proj shape: {pt_o.shape}")
            
            # Check both conversion methods
            if jax_o_converted_original.shape == pt_o.shape:
                diff_orig = np.abs(jax_o_converted_original - pt_o)
                print(f"  O proj (original conversion) max diff: {diff_orig.max():.6e}")
            
            if jax_o_converted_pali.shape == pt_o.shape:
                diff_pali = np.abs(jax_o_converted_pali - pt_o)
                print(f"  O proj (pali conversion) max diff: {diff_pali.max():.6e}")
                
            # Check what it SHOULD be (comparing to paligemma method)
            print(f"\n  DIAGNOSIS: If original conversion diff >> pali conversion diff,")
            print(f"             the o_proj conversion for gemma_expert is buggy!")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Check above for any SHAPE MISMATCH or large diff values (> 1e-5)")


def main(
    jax_checkpoint_dir: str,
    pytorch_checkpoint_dir: str,
    config_name: str = "pi05_b1k_inference_final",
    verbose: bool = False,
):
    """Compare JAX and PyTorch model weights.
    
    Args:
        jax_checkpoint_dir: Path to JAX checkpoint (e.g., /workspace/openpi/outputs/checkpoints/.../9000)
        pytorch_checkpoint_dir: Path to PyTorch checkpoint (e.g., /workspace/RLinf/safetensors_ckpts/...)
        config_name: Training config name
        verbose: Print more details
    """
    compare_weights(jax_checkpoint_dir, pytorch_checkpoint_dir, config_name, verbose)


if __name__ == "__main__":
    tyro.cli(main)
