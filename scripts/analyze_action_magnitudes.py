"""Analyze action magnitudes in your dataset to diagnose arm/navigation imbalance.

This script helps you understand why your policy might be biased towards
navigation by showing the relative magnitudes of different action groups.

Usage:
    python scripts/analyze_action_magnitudes.py --config-name=<your_config>
    
Example:
    python scripts/analyze_action_magnitudes.py --config-name=pi0_b1k
"""

import logging
import pathlib
from typing import Any

import numpy as np
import tyro
from tqdm import tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_action_statistics(
    config: _config.TrainConfig,
    num_batches: int = 100,
    action_groups: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compute statistics about action magnitudes across different groups.
    
    Args:
        config: Training configuration
        num_batches: Number of batches to analyze
        action_groups: Dictionary mapping group names to dimension ranges
        
    Returns:
        Dictionary containing statistics for each action group
    """
    # Default action groups for B1K
    if action_groups is None:
        action_groups = {
            "base": (0, 3),
            "trunk": (3, 7),
            "left_arm": (7, 14),
            "right_arm": (14, 21),
            "grippers": (21, 23),
        }
    
    # Create data loader
    logger.info("Creating data loader...")
    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=num_batches,
        framework="jax",
    )
    
    # Collect statistics
    logger.info(f"Analyzing {num_batches} batches...")
    all_stats = {name: {"magnitudes": [], "deltas": []} for name in action_groups}
    
    for batch_idx, (observation, actions) in enumerate(tqdm(data_loader, total=num_batches)):
        if batch_idx >= num_batches:
            break
            
        # Convert to numpy
        actions_np = np.array(actions)  # [B, H, D]
        
        # Compute delta actions
        delta_actions = actions_np[:, 1:, :] - actions_np[:, :-1, :]  # [B, H-1, D]
        
        # Analyze each group
        for group_name, (start_idx, end_idx) in action_groups.items():
            if start_idx >= actions_np.shape[-1]:
                continue
            end_idx = min(end_idx, actions_np.shape[-1])
            
            # Extract group actions
            group_actions = actions_np[:, :, start_idx:end_idx]
            group_deltas = delta_actions[:, :, start_idx:end_idx]
            
            # Compute magnitudes
            action_magnitudes = np.linalg.norm(group_actions, axis=-1)  # [B, H]
            delta_magnitudes = np.linalg.norm(group_deltas, axis=-1)   # [B, H-1]
            
            all_stats[group_name]["magnitudes"].extend(action_magnitudes.flatten())
            all_stats[group_name]["deltas"].extend(delta_magnitudes.flatten())
    
    # Compute summary statistics
    results = {}
    logger.info("\n" + "="*80)
    logger.info("ACTION MAGNITUDE ANALYSIS")
    logger.info("="*80)
    
    for group_name in action_groups:
        if not all_stats[group_name]["magnitudes"]:
            continue
            
        magnitudes = np.array(all_stats[group_name]["magnitudes"])
        deltas = np.array(all_stats[group_name]["deltas"])
        
        results[group_name] = {
            "magnitude_mean": float(np.mean(magnitudes)),
            "magnitude_std": float(np.std(magnitudes)),
            "magnitude_median": float(np.median(magnitudes)),
            "magnitude_p95": float(np.percentile(magnitudes, 95)),
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "delta_median": float(np.median(deltas)),
            "delta_p95": float(np.percentile(deltas, 95)),
        }
        
        logger.info(f"\n{group_name.upper()}:")
        logger.info(f"  Action magnitude: mean={results[group_name]['magnitude_mean']:.4f}, "
                   f"std={results[group_name]['magnitude_std']:.4f}, "
                   f"median={results[group_name]['magnitude_median']:.4f}, "
                   f"p95={results[group_name]['magnitude_p95']:.4f}")
        logger.info(f"  Delta magnitude:  mean={results[group_name]['delta_mean']:.4f}, "
                   f"std={results[group_name]['delta_std']:.4f}, "
                   f"median={results[group_name]['delta_median']:.4f}, "
                   f"p95={results[group_name]['delta_p95']:.4f}")
    
    # Compute relative importance (based on delta magnitude)
    logger.info("\n" + "="*80)
    logger.info("RELATIVE IMPORTANCE (based on delta magnitude)")
    logger.info("="*80)
    
    base_delta_mean = results.get("base", {}).get("delta_mean", 1.0)
    
    logger.info("\nSuggested group_weights for balanced learning:")
    logger.info("{")
    for group_name in action_groups:
        if group_name not in results:
            continue
        delta_mean = results[group_name]["delta_mean"]
        # Compute weight that would equalize contribution
        suggested_weight = base_delta_mean / (delta_mean + 1e-8)
        logger.info(f'    "{group_name}": {suggested_weight:.2f},  # delta_mean={delta_mean:.4f}')
    logger.info("}")
    
    logger.info("\nFor arm-focused training, multiply arm weights by 2-5x:")
    logger.info("{")
    for group_name in action_groups:
        if group_name not in results:
            continue
        delta_mean = results[group_name]["delta_mean"]
        suggested_weight = base_delta_mean / (delta_mean + 1e-8)
        
        # Boost arm weights
        if "arm" in group_name.lower():
            suggested_weight *= 3.0
            logger.info(f'    "{group_name}": {suggested_weight:.2f},  # 3x boosted')
        else:
            logger.info(f'    "{group_name}": {suggested_weight:.2f},')
    logger.info("}")
    
    return results


def main(config_name: str = "pi0_b1k"):
    """Analyze action magnitudes in dataset.
    
    Args:
        config_name: Name of the training config to use
    """
    # Load config
    config = _config.get_config(config_name)
    
    # Analyze
    results = compute_action_statistics(config)
    
    # Save results
    output_dir = pathlib.Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"action_magnitudes_{config_name}.json"
    
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)

